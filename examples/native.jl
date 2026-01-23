# example re-using most of Julia's native compiler functionality:
# - methods use Julia source and IR
# - inference is used to track dependencies
# - LLVM IR is generated and plugged back into Julia's JIT
# - overlay method tables are used to demonstrate method overrides
# - cache views created on-the-fly before compilation

include("julia.jl")

using Base: get_world_counter

using Base.Experimental: @MethodTable, @overlay
@MethodTable CUSTOM_MT


## Results struct for native compilation

mutable struct NativeResults
    code::Any         # (ir_bytes, entry_name) from julia_codegen
    executable::Any   # Ptr{Cvoid} from julia_jit
    NativeResults() = new(nothing, nothing)
end


## abstract interpreter

struct CustomInterpreter <: CC.AbstractInterpreter
    world::UInt
    cache::CacheView
    method_table::CC.OverlayMethodTable
    inf_cache::Vector{CC.InferenceResult}
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams

    function CustomInterpreter(cache::CacheView)
        @assert cache.world <= get_world_counter()
        new(cache.world, cache,
            CC.OverlayMethodTable(cache.world, CUSTOM_MT),
            Vector{CC.InferenceResult}(),
            CC.InferenceParams(),
            CC.OptimizationParams()
        )
    end
end

# required AbstractInterpreter interface implementation
CC.InferenceParams(interp::CustomInterpreter) = interp.inf_params
CC.OptimizationParams(interp::CustomInterpreter) = interp.opt_params
CC.get_inference_cache(interp::CustomInterpreter) = interp.inf_cache
@static if isdefined(CC, :get_inference_world)
    CC.get_inference_world(interp::CustomInterpreter) = interp.world
else
    CC.get_world_counter(interp::CustomInterpreter) = interp.world
end
CC.lock_mi_inference(::CustomInterpreter, ::Core.MethodInstance) = nothing
CC.unlock_mi_inference(::CustomInterpreter, ::Core.MethodInstance) = nothing

# Use overlay method table for method lookup during inference
CC.method_table(interp::CustomInterpreter) = interp.method_table

# integration with CompilerCaching.jl
@setup_caching CustomInterpreter.cache


## high-level API

const compilations = Ref(0) # for testing

function compile!(cache::CacheView, mi::Core.MethodInstance)
    # Get a CI through inference
    ci = get(cache, mi, nothing)
    if ci === nothing
        interp = CustomInterpreter(cache)
        CompilerCaching.typeinf!(cache, interp, mi)
        ci = get(cache, mi)
    end

    # Check for a cache hit
    res = results(cache, ci)
    if res.executable !== nothing
        return res.executable
    end
    compilations[] += 1

    # emit code: generate LLVM IR
    if res.code === nothing
        res.code = julia_codegen(cache, mi, ci)
    end

    # emit executable: JIT compile to function pointer
    if res.executable === nothing
        res.executable = julia_jit(cache, mi, res.code)
    end

    return res.executable::Ptr{Cvoid}
end

function call_generator(world::UInt, source, self, f, args)
    @nospecialize

    stub = Core.GeneratedFunctionStub(identity, Core.svec(:call, :f, :args), Core.svec())
    sig = Tuple{f, args...}

    # TODO: Use findsup and StackedMethodTable?
    mi = @ccall jl_method_lookup_by_tt(sig::Any, world::Csize_t, CUSTOM_MT::Any)::Any
    if mi === nothing
        mi = @ccall jl_method_lookup_by_tt(sig::Any, world::Csize_t, nothing::Any)::Any
    end
    if mi === nothing
        return stub(world, source, :(throw(MethodError(f, args, $world))))
    end
    mi = mi::Core.MethodInstance

    cache = CacheView{NativeResults}(:NativeExample, world)
    ci = get(cache, mi, nothing)
    if ci === nothing
        interp = CustomInterpreter(cache)
        CompilerCaching.typeinf!(cache, interp, mi)
        ci = get(cache, mi)
    end
    rettype = ci.rettype

    # Build tuple expression for ccall: (T1, T2, ...)
    ccall_types = Expr(:tuple)
    for i in 1:length(args)
        push!(ccall_types.args, args[i])
    end

    # Build argument expressions
    argexprs = Expr[]
    for i in 1:length(args)
        push!(argexprs, :(args[$i]))
    end

    ir = quote
        cache = CacheView{NativeResults}(:NativeExample, get_world_counter())
        ptr = compile!(cache, $mi)
        ccall(ptr, $rettype, $ccall_types, $(argexprs...))
    end
    # TODO: On v1.11/v1.10 stub() returns an expr and not a CodeInfo
    # see create_fresh_codeinfo in Enzyme.jl, but we must lower the ir above
    # ourselves...
    code_info = stub(world, source, ir)
    if code_info isa CC.CodeInfo
        code_info.edges = Any[mi]
    end
    return code_info
end

@eval function call(f, args...)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, call_generator))
end


## demo

# Define `op` with different implementations
op(x, y) = x + y
@overlay CUSTOM_MT op(x, y) = x * y

parent(x) = child(x) + 1
child(x) = op(x, 2)

# Test whether overlay is working
@assert parent(10) == 10 + 2 + 1
@assert call(parent, 10) == 10 * 2 + 1

# Ensure we don't needlessly recompile on repeated or and unrelated definitions
@assert compilations[] == 1
call(parent, 10)
unrelated(x) = 10
call(parent, 10)
@assert compilations[] == 1

# Redefine parent function
parent(x) = child(x) + 3
@assert parent(10) == 10 + 2 + 3
@assert call(parent, 10) == 10 * 2 + 3
@assert compilations[] == 2

# Redefine child function
child(x) = op(x, 3)
@assert parent(10) == 10 + 3 + 3
@assert call(parent, 10) == 10 * 3 + 3
@assert compilations[] == 3

# Redefine overlay operator
@eval @overlay CUSTOM_MT op(x, y) = x ^ y
@assert parent(10) == 10 + 3 + 3
@assert call(parent, 10) == 10 ^ 3 + 3
@assert compilations[] == 4
