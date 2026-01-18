# examples/julia.jl - Julia native code compilation helpers

using CompilerCaching
using LLVM
using LLVM.Interop

const CC = Core.Compiler
using Base: get_world_counter, CodegenParams

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
    julia_codegen(cache, mi, world, codeinfos) -> (ir_bytes, entry_name)

Generate LLVM IR and return serializable intermediate result.
Returns a tuple of (LLVM bitcode bytes, entry function name).

On Julia 1.12+, uses `codeinfos` (CodeInstance/CodeInfo pairs) for codegen.
On Julia 1.11, `codeinfos` contains `[ci => nothing]`; uses cache lookup callback instead.

This function handles codegen but does not JIT compile - use `julia_jit` for that.
"""
function julia_codegen(cache::CompilerCache, mi::Core.MethodInstance,
                       world::UInt, codeinfos::Vector{<:Pair{Core.CodeInstance}})

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

    # Get JuliaOJIT for target configuration
    jljit = getglobal_jljit()

    # Generate LLVM IR
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

        # Get the ThreadSafeModule
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

        # Serialize to bitcode
        ir_bytes = llvm_ts_mod() do mod
            convert(Vector{UInt8}, mod)
        end

        return (ir_bytes, func_name)
    end
end

"""
    julia_jit(cache, mi, world, ir_data) -> Ptr{Cvoid}

JIT compile LLVM bitcode to a function pointer.

Takes a tuple of (LLVM bitcode bytes, entry function name) as returned by `julia_codegen`.
The `cache`, `mi`, and `world` arguments are ignored but included for use as a `link` callback.
"""
function julia_jit(cache, mi, world, ir_data)
    ir_bytes, entry_name = ir_data

    jljit = getglobal_jljit()
    jd = JITDylib(jljit)

    with_llvm_context() do ctx
        # Parse bitcode back into a module, then wrap in ThreadSafeModule
        mod = parse(LLVM.Module, ir_bytes)
        ts_mod = ThreadSafeModule(mod)

        # Run Julia's optimization pipeline to lower intrinsics
        ts_mod() do m
            run!(JuliaPipeline(), m)
        end

        # Add to JIT
        add!(jljit, jd, ts_mod)

        # Look up the compiled function
        addr = LLVM.lookup(jljit, entry_name)
        return pointer(addr)
    end
end
