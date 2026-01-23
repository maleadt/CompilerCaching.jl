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
    ir::Any           # Vector{Pair{CodeInstance, CodeInfo}} from typeinf!
    code::Any         # (ir_bytes, entry_name) from julia_codegen
    executable::Any   # Ptr{Cvoid} from julia_jit
    NativeResults() = new(nothing, nothing, nothing)
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
    # For inference-based compilation, first check if CI exists
    ci = get(cache, mi, nothing)

    if ci !== nothing
        res = results(cache, ci)
        if res.executable !== nothing
            return res.executable
        end
    end

    # Cache miss or incomplete - need to compile

    # emit_ir: runs inference (this creates the CI)
    interp = CustomInterpreter(cache)
    ir = CompilerCaching.typeinf!(cache, interp, mi)

    # Get the CI that was created by typeinf!
    ci = get(cache, mi, nothing)
    @assert ci !== nothing "typeinf! should have created a CodeInstance"
    res = results(cache, ci)
    res.ir = ir

    # emit_code: generates LLVM IR
    compilations[] += 1
    res.code = julia_codegen(cache, mi, res.ir)

    # emit_executable: JIT compiles to native code
    res.executable = julia_jit(cache, mi, res.code)

    return res.executable
end

"""
    call(f, args...) -> result

Compile (if needed) and call function `f` with the given arguments.
"""
@inline function call(f, args...)
    argtypes = Tuple{map(Core.Typeof, args)...}
    rettyp = Base.infer_return_type(f, argtypes)
    _call_impl(rettyp, f, args...)
end
@generated function _call_impl(::Type{R}, f, args::Vararg{Any,N}) where {R,N}
    argtypes = Tuple{args...}

    # Build tuple expression for ccall: (T1, T2, ...)
    ccall_types = Expr(:tuple)
    for i in 1:N
        push!(ccall_types.args, args[i])
    end

    # Build argument expressions
    argexprs = Expr[]
    for i in 1:N
        push!(argexprs, :(args[$i]))
    end

    quote
        world = get_world_counter()
        mi = @something(method_instance(f, $argtypes; world, method_table=CUSTOM_MT),
                        method_instance(f, $argtypes; world),
                        throw(MethodError(f, $argtypes)))

        cache = CacheView{NativeResults}(:NativeExample, world)
        ptr = compile!(cache, mi)
        ccall(ptr, R, $ccall_types, $(argexprs...))
    end
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
