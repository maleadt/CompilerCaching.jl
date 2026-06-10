# EXCLUDE FROM TESTING
#
# @comptime example — compile-time evaluation via OpaqueClosure
#
# Julia's native compiler already handles constant propagation for literals,
# so `sin(1)` or `f(x, 3)` get optimized automatically. @comptime adds value
# when a value is constant at a call site but NOT visible as a literal to the
# compiler — e.g. an op-code read from a table, or a config loaded at startup.
#
# Demonstrates const-seeded inference with two execution paths:
# - Fully-const calls: extract rettype_const directly (zero cost)
# - Partially-const calls: wrap const-optimized CodeInfo in OpaqueClosure
#
# No LLVM.jl dependency needed — uses Julia's native calling convention.

using CompilerCaching
using CompilerCaching: get_source, SpecializedResult, CachedResult
const CC = Core.Compiler
using Base: get_world_counter


## Results struct

mutable struct ComptimeResults
    oc::Any       # cached OpaqueClosure (partially-const path)
    val::Any      # cached rettype_const (fully-const path)
    ComptimeResults() = new(nothing, nothing)
end


## Abstract interpreter (no overlay method table — simpler than native.jl)

const InfCacheT = @static if isdefined(CC, :InferenceCache)
    CC.InferenceCache
else
    Vector{CC.InferenceResult}
end

struct ComptimeInterpreter <: CC.AbstractInterpreter
    world::UInt
    cache::CacheView
    inf_cache::InfCacheT
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams

    function ComptimeInterpreter(cache::CacheView)
        @assert cache.world <= get_world_counter()
        new(cache.world, cache, InfCacheT(),
            CC.InferenceParams(), CC.OptimizationParams())
    end
end

CC.InferenceParams(interp::ComptimeInterpreter) = interp.inf_params
CC.OptimizationParams(interp::ComptimeInterpreter) = interp.opt_params
CC.get_inference_cache(interp::ComptimeInterpreter) = interp.inf_cache
@static if isdefined(CC, :get_inference_world)
    CC.get_inference_world(interp::ComptimeInterpreter) = interp.world
else
    CC.get_world_counter(interp::ComptimeInterpreter) = interp.world
end
CC.lock_mi_inference(::ComptimeInterpreter, ::Core.MethodInstance) = nothing
CC.unlock_mi_inference(::ComptimeInterpreter, ::Core.MethodInstance) = nothing

CC.cache_owner(interp::ComptimeInterpreter) = interp.cache.owner


## Helper: get SpecializedResult for given argtypes

function get_specialized_result(ci::Core.CodeInstance, argtypes::Vector{Any})
    cached = CC.traverse_analysis_results(ci) do @nospecialize result
        result isa CachedResult{ComptimeResults} ? result : nothing
    end
    cached === nothing && return nothing
    for entry in cached.const_entries
        entry.argtypes == argtypes && return entry
    end
    return nothing
end


## Core compilation logic

const compilations = Ref(0)

function compile!(cache::CacheView, mi::Core.MethodInstance, argtypes::Vector{Any})
    # Ensure generic CI exists
    ci = get(cache, mi, nothing)
    if ci === nothing
        interp = ComptimeInterpreter(cache)
        ci = CompilerCaching.typeinf!(interp, mi)
    end

    # Run const-seeded inference (cached/idempotent)
    entry = get_specialized_result(ci, argtypes)
    if entry === nothing
        interp = ComptimeInterpreter(cache)
        CompilerCaching.typeinf!(cache, interp, mi, argtypes)
        entry = get_specialized_result(ci, argtypes)
        @assert entry !== nothing "const-seeded inference failed for $mi with argtypes $argtypes"
    end

    # Check for cached result
    res = entry.inner::ComptimeResults
    res.val !== nothing && return res.val
    res.oc !== nothing && return res.oc

    compilations[] += 1

    # Fully-const path: inference proved a constant return value
    if entry.rettype_const !== nothing
        res.val = entry.rettype_const
        return res.val
    end

    # Partially-const path: build OpaqueClosure from const-optimized CodeInfo
    src = entry.src::Core.CodeInfo
    rettype = CC.widenconst(entry.rettype)
    nargs = Int(mi.def.nargs) - 1   # user args, excluding function self
    sig = Tuple{mi.def.sig.parameters[2:end]...}

    # Fix slot 1: CodeInfo has the original function type, but OC expects its own self type.
    # Const-seeded inference folds away all references to the function arg, so this is safe.
    src.slottypes[1] = Any

    oc = Core.OpaqueClosure(src; rettype, nargs, sig)
    res.oc = oc
    return oc
end


## @comptime macro

"""
    @comptime f(args...)

Compile-time evaluation macro. Wrap arguments in `Core.Const(val)` to mark them
as compile-time constants. The compiler will specialize on these values:

- If ALL arguments are `Core.Const`, the result is computed at compile time.
- If SOME arguments are `Core.Const`, the compiler eliminates dead branches and
  returns an OpaqueClosure that executes the optimized code.

# Examples
```julia
@comptime evaluate(Core.Const(2), 10, 20) # const op → dead branches eliminated
@comptime kernel(42, Core.Const(256))      # const config → single branch remains
```
"""
macro comptime(ex)
    ex isa Expr && ex.head === :call || error("@comptime requires a function call, got: $ex")

    f = ex.args[1]
    call_args = ex.args[2:end]

    # Detect Core.Const(val) pattern in AST
    _is_core_const(x) = x isa Expr && x.head === :. &&
        length(x.args) == 2 && x.args[1] === :Core &&
        x.args[2] isa QuoteNode && x.args[2].value === :Const

    # Build mask: true where argument is Core.Const(val)
    mask = Bool[]
    unwrapped_args = Any[]
    for arg in call_args
        if arg isa Expr && arg.head === :call && _is_core_const(arg.args[1])
            push!(mask, true)
            push!(unwrapped_args, arg.args[2])  # unwrap the value
        else
            push!(mask, false)
            push!(unwrapped_args, arg)
        end
    end

    mask_tuple = Expr(:tuple, mask...)
    args_tuple = Expr(:tuple, (esc(a) for a in unwrapped_args)...)

    return quote
        _comptime_call($(esc(f)), Val($(mask_tuple)), $(args_tuple))
    end
end

function _comptime_call(f, ::Val{mask}, args::Tuple) where {mask}
    # Build argtypes with Core.Const for const positions
    argtypes = Any[Core.Const(f)]
    for i in 1:length(args)
        if mask[i]
            push!(argtypes, Core.Const(args[i]))
        else
            push!(argtypes, Core.Typeof(args[i]))
        end
    end

    # Look up MethodInstance
    tt = Tuple{map(Core.Typeof, args)...}
    world = get_world_counter()
    mi = method_instance(f, tt; world)
    mi === nothing && throw(MethodError(f, args))

    # Compile
    cache = CacheView{ComptimeResults}(:ComptimeExample, world)
    result = compile!(cache, mi, argtypes)

    # Dispatch on result type
    if result isa Core.OpaqueClosure
        return result(args...)
    else
        return result
    end
end


## Demo 1: Operation dispatch elimination
#
# evaluate() dispatches on an integer op code — a pattern common in interpreters,
# DSL compilers, and GPU shader compilers. The op is known at each call site but
# stored in a variable, so Julia's native compiler keeps all branches.

println("=== Demo 1: Operation dispatch elimination ===")

function evaluate(op::Int, x, y)
    if op == 1;     return x + y
    elseif op == 2; return x * y
    elseif op == 3; return max(x, y)
    else;           return zero(x)
    end
end

result = @comptime evaluate(Core.Const(2), 10, 20)
println("@comptime evaluate(Core.Const(2), 10, 20) = $result")
@assert result == 200 "Expected 200, got $result"
@assert compilations[] == 1
println("Compilations: $(compilations[])")


## Demo 2: Configuration-driven specialization
#
# Runtime config parameter — known after init, constant during execution.
# Julia's compiler can't see through Ref, so all branches remain.

println("\n=== Demo 2: Configuration-driven specialization ===")

function kernel(x, block_size)
    if block_size <= 128
        x * 2
    elseif block_size <= 256
        x * 3
    else
        x * 4
    end
end

result = @comptime kernel(42, Core.Const(256))
println("@comptime kernel(42, Core.Const(256)) = $result")
@assert result == 126 "Expected 126, got $result"
@assert compilations[] == 2
println("Compilations: $(compilations[])")


## Demo 3: Verify optimization (compare generic vs const-seeded CodeInfo)

println("\n=== Demo 3: Verify branch elimination ===")

let
    world = get_world_counter()
    cache = CacheView{ComptimeResults}(:ComptimeExample, world)
    mi = method_instance(evaluate, (Int, Int, Int); world)
    ci = get(cache, mi)

    # Generic CodeInfo should have GotoIfNot (branches)
    generic_src = get_source(ci)::Core.CodeInfo
    has_branch = any(x -> x isa Core.GotoIfNot, generic_src.code)
    generic_stmts = length(generic_src.code)
    println("Generic CodeInfo: $generic_stmts stmts, has branch: $has_branch")
    @assert has_branch "Generic CodeInfo should contain GotoIfNot"

    # Const-seeded CodeInfo (op=2) should NOT have GotoIfNot
    argtypes = Any[Core.Const(evaluate), Core.Const(2), Int, Int]
    const_src = get_source(ci, argtypes)::Core.CodeInfo
    has_branch_const = any(x -> x isa Core.GotoIfNot, const_src.code)
    const_stmts = length(const_src.code)
    println("Const-seeded CodeInfo (op=2): $const_stmts stmts, has branch: $has_branch_const")
    @assert !has_branch_const "Const-seeded CodeInfo should eliminate GotoIfNot"

    println("Generic code:       $(generic_src.code)")
    println("Const-seeded code:  $(const_src.code)")
end


## Demo 4: Caching

println("\n=== Demo 4: Caching ===")
compilations[] = 0

result1 = @comptime evaluate(Core.Const(2), 10, 20)
println("evaluate(op=2, 10, 20) again: result=$result1, compilations=$(compilations[])")
@assert result1 == 200
@assert compilations[] == 0 "Should be cached (got $(compilations[]) compilations)"

result2 = @comptime kernel(42, Core.Const(256))
println("kernel(42, bs=256) again: result=$result2, compilations=$(compilations[])")
@assert result2 == 126
@assert compilations[] == 0 "Should be cached (got $(compilations[]) compilations)"

println("\nAll assertions passed!")
