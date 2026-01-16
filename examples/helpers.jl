# examples/helpers.jl - Reusable compilation helpers
#
# Provides inference and native code emission utilities that can be used by
# different example compilers. Each example defines its own AbstractInterpreter.
#
# Requires Julia 1.11+ and LLVM.jl

using CompilerCaching
using LLVM
using LLVM.Interop

const CC = Core.Compiler
using Base: get_world_counter, CodegenParams

#==============================================================================#
# Inference
#==============================================================================#

"""
    run_inference(cache, interp, mi) -> Union{Vector{Pair{CodeInstance, CodeInfo}}, Nothing}

Run type inference on `mi` using the provided interpreter.

The interpreter must implement the `CC.AbstractInterpreter` interface.

On Julia 1.12+, returns a vector of (CodeInstance, CodeInfo) pairs for codegen.
On Julia 1.11, returns `nothing` (cache is populated implicitly by typeinf_ext_toplevel).
"""
function run_inference(cache::CompilerCache, interp::CC.AbstractInterpreter,
                       mi::Core.MethodInstance)
    @static if VERSION >= v"1.12.0-DEV.1434"
        # Modern API: returns CodeInstance, use SOURCE_MODE to control caching
        ci = CC.typeinf_ext(interp, mi, CC.SOURCE_MODE_FORCE_SOURCE)
        @assert ci !== nothing "Inference of $mi failed"

        # Collect all code that needs compilation (including callees)
        codeinfos = Pair{Core.CodeInstance, Core.CodeInfo}[]

        @static if VERSION >= v"1.13.0-DEV.499" || v"1.12-beta3" <= VERSION < v"1.13-"
            workqueue = CC.CompilationQueue(; interp)
            push!(workqueue, ci)
            while !isempty(workqueue)
                callee = pop!(workqueue)
                CC.isinspected(workqueue, callee) && continue
                CC.markinspected!(workqueue, callee)

                callee_mi = CC.get_ci_mi(callee)
                if CC.use_const_api(callee)
                    @static if VERSION >= v"1.13.0-DEV.1121"
                        src = CC.codeinfo_for_const(interp, callee_mi,
                            CC.WorldRange(callee.min_world, callee.max_world),
                            callee.edges, callee.rettype_const)
                    else
                        src = CC.codeinfo_for_const(interp, callee_mi, callee.rettype_const)
                    end
                else
                    src = CC.typeinf_code(interp, callee_mi, true)
                end
                if src isa Core.CodeInfo
                    sptypes = CC.sptypes_from_meth_instance(callee_mi)
                    CC.collectinvokes!(workqueue, src, sptypes)
                    push!(codeinfos, callee => src)
                end
            end
        else
            # Older 1.12 API
            workqueue = Core.CodeInstance[ci]
            inspected = IdSet{Core.CodeInstance}()
            while !isempty(workqueue)
                callee = pop!(workqueue)
                callee in inspected && continue
                push!(inspected, callee)

                callee_mi = CC.get_ci_mi(callee)
                if CC.use_const_api(callee)
                    src = CC.codeinfo_for_const(interp, callee_mi, callee.rettype_const)
                else
                    src = CC.typeinf_code(interp, callee_mi, true)
                end
                if src isa Core.CodeInfo
                    CC.collectinvokes!(workqueue, src)
                    push!(codeinfos, callee => src)
                end
            end
        end

        return codeinfos
    elseif VERSION >= v"1.12.0-DEV.15"
        # Julia 1.12 early API
        ci = CC.typeinf_ext_toplevel(interp, mi, CC.SOURCE_MODE_FORCE_SOURCE)
        @assert ci !== nothing "Inference of $mi failed"
        return Pair{Core.CodeInstance, Core.CodeInfo}[]
    else
        # Julia 1.11 API - cache populated implicitly, return nothing
        src = CC.typeinf_ext_toplevel(interp, mi)

        # Handle const-return case where ci.inferred may be nothing
        world = @static if isdefined(CC, :get_inference_world)
            CC.get_inference_world(interp)
        else
            CC.get_world_counter(interp)
        end
        ci = CompilerCaching.lookup(cache, mi; world)
        if ci !== nothing && ci.inferred === nothing
            @atomic ci.inferred = src
        end

        return nothing
    end
end

#==============================================================================#
# Native Code Emission
#==============================================================================#

# LLVM context helper
function with_llvm_context(f)
    ts_ctx = ThreadSafeContext()
    ctx = context(ts_ctx)
    activate(ctx)
    try
        f(ctx)
    finally
        deactivate(ctx)
        dispose(ts_ctx)
    end
end

# Callback function for codegen to look in the cache
const _codegen_cache = Ref{Any}(nothing)
const _codegen_world = Ref{UInt}(0)
function _codegen_lookup_cb(mi, min_world, max_world)
    ci = CompilerCaching.lookup(_codegen_cache[], mi; world=min_world)
    @static if VERSION < v"1.12.0-DEV.1434"
        # Refuse to return CI without source - force re-inference before codegen
        if ci !== nothing && ci.inferred === nothing
            return nothing
        end
    end
    return ci
end

# Global JuliaOJIT instance
const global_jljit = Ref{Any}(nothing)
function getglobal_jljit()
    if global_jljit[] === nothing
        jljit = JuliaOJIT()
        # Add process symbol generator so Julia runtime symbols can be resolved
        jd = JITDylib(jljit)
        prefix = LLVM.get_prefix(jljit)
        dg = LLVM.CreateDynamicLibrarySearchGeneratorForProcess(prefix)
        add!(jd, dg)
        global_jljit[] = jljit
    end
    return global_jljit[]
end

"""
    emit_native(cache, interp, mi, codeinfos) -> Ptr{Cvoid}

Generate native code for `mi` and JIT compile it.
Returns a function pointer to the compiled code.

On Julia 1.12+, uses `codeinfos` (CodeInstance/CodeInfo pairs) for codegen.
On Julia 1.11, `codeinfos` should be `nothing`; uses cache lookup callback instead.

The `cache` is used to look up CodeInstances during codegen.
"""
function emit_native(cache::CompilerCache, interp::CC.AbstractInterpreter,
                     mi::Core.MethodInstance,
                     codeinfos::Union{Vector{Pair{Core.CodeInstance, Core.CodeInfo}}, Nothing})
    # Extract world from interpreter
    world = @static if isdefined(CC, :get_inference_world)
        CC.get_inference_world(interp)
    else
        CC.get_world_counter(interp)
    end

    # Set up globals for the lookup callback
    _codegen_cache[] = cache
    _codegen_world[] = world
    lookup_cfunction = @cfunction(_codegen_lookup_cb, Any, (Any, UInt, UInt))

    # Set up codegen parameters
    @static if VERSION < v"1.12.0-DEV.1667"
        params = CodegenParams(; lookup = Base.unsafe_convert(Ptr{Nothing}, lookup_cfunction))
    else
        params = CodegenParams()
    end

    # Get JuliaOJIT for JIT compilation
    jljit = getglobal_jljit()
    jd = JITDylib(jljit)

    # Generate LLVM IR and JIT compile
    with_llvm_context() do ctx
        # Create LLVM module
        ts_mod = ThreadSafeModule("native_compile")

        # Configure module for native target using JuliaOJIT's settings
        ts_mod() do mod
            triple!(mod, triple(jljit))
            datalayout!(mod, datalayout(jljit))
        end

        # Generate native code
        @static if VERSION >= v"1.12.0-DEV.1823"
            cis_vec = Any[]
            for (ci, src) in codeinfos
                push!(cis_vec, ci)
                push!(cis_vec, src)
            end
            native_code = @ccall jl_emit_native(
                cis_vec::Vector{Any},
                ts_mod::LLVM.API.LLVMOrcThreadSafeModuleRef,
                Ref(params)::Ptr{CodegenParams},
                false::Cint
            )::Ptr{Cvoid}
        elseif VERSION >= v"1.12.0-DEV.1667"
            native_code = @ccall jl_create_native(
                [mi]::Vector{Core.MethodInstance},
                ts_mod::LLVM.API.LLVMOrcThreadSafeModuleRef,
                Ref(params)::Ptr{CodegenParams},
                1::Cint, 0::Cint, 0::Cint, world::Csize_t,
                lookup_cfunction::Ptr{Cvoid}
            )::Ptr{Cvoid}
        else
            native_code = @ccall jl_create_native(
                [mi]::Vector{Core.MethodInstance},
                ts_mod::LLVM.API.LLVMOrcThreadSafeModuleRef,
                Ref(params)::Ptr{CodegenParams},
                1::Cint, 0::Cint, 0::Cint, world::Csize_t
            )::Ptr{Cvoid}
        end

        @assert native_code != C_NULL "Code generation failed"

        # Get the ThreadSafeModule for JIT
        llvm_mod_ref = @ccall jl_get_llvm_module(
                native_code::Ptr{Cvoid}
            )::LLVM.API.LLVMOrcThreadSafeModuleRef
        @assert llvm_mod_ref != C_NULL "Failed to get LLVM module"

        llvm_ts_mod = ThreadSafeModule(llvm_mod_ref)

        # Get function name from CodeInstance
        ci = CompilerCaching.lookup(cache, mi; world)
        @assert ci !== nothing "CodeInstance not found after codegen"

        func_idx = Ref{Int32}(-1)
        specfunc_idx = Ref{Int32}(-1)
        @ccall jl_get_function_id(native_code::Ptr{Cvoid}, ci::Any,
               func_idx::Ptr{Int32}, specfunc_idx::Ptr{Int32})::Nothing

        func_name = nothing
        if specfunc_idx[] >= 1
            func_ref = @ccall jl_get_llvm_function(
                    native_code::Ptr{Cvoid},
                    (specfunc_idx[] - 1)::UInt32
                )::LLVM.API.LLVMValueRef
            @assert func_ref != C_NULL
            func_name = name(LLVM.Function(func_ref))
        elseif func_idx[] >= 1
            func_ref = @ccall jl_get_llvm_function(
                    native_code::Ptr{Cvoid},
                    (func_idx[] - 1)::UInt32
                )::LLVM.API.LLVMValueRef
            @assert func_ref != C_NULL
            func_name = name(LLVM.Function(func_ref))
        end

        @assert func_name !== nothing "No compiled function found"

        # Run Julia's optimization pipeline to lower intrinsics
        llvm_ts_mod() do mod
            run!(JuliaPipeline(), mod)
        end

        # Add to JIT
        add!(jljit, jd, llvm_ts_mod)

        # Look up the compiled function
        addr = LLVM.lookup(jljit, func_name)
        return pointer(addr)
    end
end

#==============================================================================#
# Utilities
#==============================================================================#

"""
    show_edges(ci::CodeInstance)

Print the edges (dependencies) of a CodeInstance for debugging.
"""
function show_edges(ci::Core.CodeInstance)
    if isdefined(ci, :edges) && ci.edges !== nothing
        edges = ci.edges
        if edges isa Core.SimpleVector && length(edges) > 0
            println("   Edges ($(length(edges)) dependencies):")
            for (i, edge) in enumerate(edges)
                if edge isa Core.MethodInstance
                    println("     [$i] $(edge.def.name) :: $(edge.specTypes)")
                else
                    println("     [$i] $edge")
                end
            end
        else
            println("   Edges: (none)")
        end
    else
        println("   Edges: (not defined)")
    end
end
