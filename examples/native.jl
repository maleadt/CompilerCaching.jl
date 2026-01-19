# example re-using most of Julia's native compiler functionality:
# - methods use Julia source and IR
# - inference is used to track dependencies
# - LLVM IR is generated and plugged back into Julia's JIT
# - overlay method tables are used to demonstrate method overrides

include("julia.jl")

using Base: get_world_counter

using Base.Experimental: @MethodTable, @overlay
@MethodTable CUSTOM_MT

# Enable disk caching using the three-phase API with CI build_id keys.
# This avoids stale bitcode conflicts on method redefinition.
const CUSTOM_CACHE = CompilerCache(:NativeExample, CUSTOM_MT;
                                   disk_cache=(VERSION >= v"1.12-"))


## abstract interpreter

struct CustomInterpreter <: CC.AbstractInterpreter
    world::UInt
    cache::CompilerCache
    method_table::CC.OverlayMethodTable
    inf_cache::Vector{CC.InferenceResult}
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams

    function CustomInterpreter(cache::CompilerCache, world::UInt)
        @assert world <= get_world_counter()
        new(world, cache,
            CC.OverlayMethodTable(world, CUSTOM_MT),
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

# Inference phase: returns Vector{Pair{CI, IR}} where root CI is first entry
function julia_infer(cache::CompilerCache, mi::Core.MethodInstance, world::UInt)
    interp = CustomInterpreter(cache, world)
    CompilerCaching.populate!(cache, interp, mi)
end

# Codegen phase wrapper: counts compilations for testing
const compilations = Ref(0) # for testing
function julia_codegen_counted(cache::CompilerCache, mi::Core.MethodInstance, world::UInt, codeinfos)
    compilations[] += 1
    julia_codegen(cache, mi, world, codeinfos)
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

        ptr = _call_compile(CUSTOM_CACHE, mi, world)
        ccall(ptr, R, $ccall_types, $(argexprs...))
    end
end
function _call_compile(cache, mi, world)
    cached_compilation(cache, mi, world;
        infer = julia_infer,
        codegen = julia_codegen_counted,
        link = julia_jit
    )
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
