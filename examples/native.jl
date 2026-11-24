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

const InfCacheT = @static if isdefined(CC, :InferenceCache)
    CC.InferenceCache
else
    Vector{CC.InferenceResult}
end

struct CustomInterpreter <: CC.AbstractInterpreter
    world::UInt
    cache::CacheView
    method_table::CC.OverlayMethodTable
    inf_cache::InfCacheT
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams

    function CustomInterpreter(cache::CacheView)
        @assert cache.world <= get_world_counter()
        new(cache.world, cache,
            CC.OverlayMethodTable(cache.world, CUSTOM_MT),
            InfCacheT(),
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

# integration with CompilerCaching.jl: just route the owner token to the cache view.
# Results structs are attached lazily by `results`, so no further hooks are needed.
CC.cache_owner(interp::CustomInterpreter) = interp.cache.owner


## high-level API

const compilations = Ref(0) # for testing

function compile!(cache::CacheView, mi::Core.MethodInstance)
    # Get a CI through inference
    interp = CustomInterpreter(cache)
    ci = get(cache, mi, nothing)
    if ci === nothing
        ci = CompilerCaching.typeinf!(interp, mi)
    end

    # Check for a cache hit
    res = results(cache, ci)
    if res.executable !== nothing
        return res.executable
    end
    compilations[] += 1

    # emit code: generate LLVM IR
    if res.code === nothing
        res.code = julia_codegen(cache, interp, mi, ci)
    end

    # emit executable: JIT compile to function pointer
    if res.executable === nothing
        res.executable = julia_jit(cache, mi, res.code)
    end

    return res.executable
end

function compile!(cache::CacheView, mi::Core.MethodInstance, argtypes::Vector{Any})
    interp = CustomInterpreter(cache)
    ci = get(cache, mi, nothing)
    if ci === nothing
        ci = CompilerCaching.typeinf!(interp, mi)
    end

    # Get the const-specialized entry, running const-seeded inference if needed
    entry = @something CompilerCaching.specialization(cache, ci, argtypes) begin
        CompilerCaching.typeinf!(cache, interp, mi, argtypes)
    end

    # codegen + JIT using const-optimized source
    code = julia_codegen(cache, interp, mi, ci; entry)
    return julia_jit(cache, mi, code)
end

"""
    call(f, args...) -> result

Compile (if needed) and call function `f` with the given arguments.

The signature of the compiled code is only known once it has been compiled: the return type
is the one our interpreter inferred, which can differ from what Julia infers natively (e.g.
because of an overlay method), and the argument types are those of the method instance,
which can be wider than the types of `args` (e.g. with `@nospecialize`). Both are thus
looked up on every call, making the actual invocation a dynamic dispatch that boxes its
arguments. Avoiding that requires a call with statically-known types that still observes
method redefinitions, as proposed in JuliaLang/julia#59774.
"""
function call(f, args...)
    argtypes = Tuple{map(Core.Typeof, args)...}
    world = get_world_counter()
    mi = @something(method_instance(f, argtypes; world, method_table=CUSTOM_MT),
                    method_instance(f, argtypes; world),
                    throw(MethodError(f, args, world)))

    cache = CacheView{NativeResults}(:NativeExample, world)
    ptr = compile!(cache, mi)
    ci = get(cache, mi)
    invoke_pointer(ptr, ci.rettype, mi.specTypes, args...)
end

# Only suitable for simple signatures: the pointer is called using the C calling convention,
# which matches how Julia passes and returns boxed values and primitive types.
@generated function invoke_pointer(ptr::Ptr{Cvoid}, ::Type{R}, ::Type{Sig},
                                   args::Vararg{Any,N}) where {R,Sig,N}
    # skip the function, which is the first parameter of the signature
    ccall_types = Expr(:tuple, Sig.parameters[2:end]...)
    argexprs = [:(args[$i]) for i in 1:N]
    :(ccall(ptr, R, $ccall_types, $(argexprs...)))
end

@static if VERSION >= v"1.14.0-DEV.1794"
@warn "Native example is broken on Julia 1.14"
else


## demo of overlay + redefinitions

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


## demo of signatures that differ from what the caller sees

# An overlay can change the return type
scale(x) = x * 2
@overlay CUSTOM_MT scale(x) = x * 2.5
@assert scale(10) === 20
@assert call(scale, 10) === 25.0

# Arguments that the method doesn't specialize on are passed boxed
passthrough(@nospecialize(x)) = x
@assert call(passthrough, 42) === 42


## demo of const-seeded codegen

function constdemo_f(x, n)
    if n > 0
        return x + n
    else
        return x - n
    end
end

let
    world = get_world_counter()
    cache = CacheView{NativeResults}(:NativeExample, world)
    mi = method_instance(constdemo_f, (Int, Int); world)
    argtypes = Any[Core.Const(constdemo_f), Int, Core.Const(3)]

    # Ensure inference is done
    interp = CustomInterpreter(cache)
    ci = CompilerCaching.typeinf!(interp, mi)
    entry = CompilerCaching.typeinf!(cache, interp, mi, argtypes)

    # Generic codegen
    (_, _, generic_ir) = julia_codegen(cache, interp, mi, ci; dump_llvm=true)
    println("=== Generic LLVM IR ===")
    println(generic_ir)
    @assert contains(generic_ir, "icmp") "Generic IR should have a comparison"
    @assert contains(generic_ir, "sub i64") "Generic IR should have the sub branch"

    # Const-seeded codegen (n=3 is a known constant)
    (_, _, const_ir) = julia_codegen(cache, interp, mi, ci; entry, dump_llvm=true)
    println("\n=== Const-seeded LLVM IR (n=3) ===")
    println(const_ir)
    @assert !contains(const_ir, "icmp") "Const-seeded IR should eliminate the comparison"
    @assert !contains(const_ir, "sub i64") "Const-seeded IR should eliminate the sub"
end


end
