# Const-seeded inference: every seeded root is a CodeInstance of its own, published
# under a private owner, carrying the edges and world range of the inference that
# produced its source.

const CC = Core.Compiler

mutable struct SeededResults
    tag::Any
    SeededResults() = new(nothing)
end

struct SeededInterpreter <: CC.AbstractInterpreter
    world::UInt
    owner::Any
    inf_cache::InfCacheT
    fail::Base.RefValue{Bool}
    codegen::Union{Nothing, IdDict{Core.CodeInstance, Core.CodeInfo}}
end
SeededInterpreter(cache::CacheView; fail=Ref(false), codegen=nothing) =
    SeededInterpreter(cache.world, cache.owner, InfCacheT(), fail, codegen)
CC.cache_owner(interp::SeededInterpreter) = interp.owner
CC.InferenceParams(::SeededInterpreter) = CC.InferenceParams()
CC.OptimizationParams(::SeededInterpreter) = CC.OptimizationParams()
CC.get_inference_cache(interp::SeededInterpreter) = interp.inf_cache
CC.get_inference_world(interp::SeededInterpreter) = interp.world
@static if isdefined(CC, :codegen_cache)
    CC.codegen_cache(interp::SeededInterpreter) = interp.codegen
end
CC.method_table(interp::SeededInterpreter) =
    interp.fail[] ? error("injected inference failure") :
                    CC.InternalMethodTable(interp.world)

# The Float32 literals in a source, which is how the tests below tell which
# method of a callee the seeded inference folded in.
float_consts(src::Core.CodeInfo) =
    sort!(unique(x for st in src.code if st isa Expr for x in st.args if x isa Float32))

# All CodeInstances of `mi` with the given owner, valid or not.
function owned_cis(mi::Core.MethodInstance, @nospecialize(owner))
    cis = Core.CodeInstance[]
    isdefined(mi, :cache) || return cis
    ci = mi.cache
    while true
        ci.owner === owner && push!(cis, ci)
        isdefined(ci, :next) || break
        ci = ci.next
    end
    return cis
end

# Julia 1.14 without root inference under a separate owner does not extend a seeded
# result to later worlds (see `typeinf!`).
const seeds_extended = isdefined(CC, :CACHE_MODE_VOLATILE) || isdefined(CC, :CACHE_MODE_ROOT)

seeded_add(a, b) = a + b
seeded_sub(a, b) = a - b
seeded_va(a, b...) = +(a, b...)

@testset "const-seeded inference" begin
    world = Base.get_world_counter()
    cache = CacheView{SeededResults}(:SeededTest, world)
    interp = SeededInterpreter(cache)
    mi = method_instance(seeded_add, (Int, Int); world)

    argtypes = Any[CC.Const(seeded_add), CC.Const(1), CC.Const(2)]
    view = seeded(cache, argtypes)
    @test get(view, mi, nothing) === nothing

    ci = typeinf!(interp, mi, argtypes)
    @test ci isa Core.CodeInstance
    @test ci.owner === view.owner
    @test ci.rettype === Int
    @test ci.rettype_const === 3
    @test get_source(ci) isa Core.CodeInfo

    # found through the seeded view, with results attached to the seeded CI
    @test get(view, mi, nothing) === ci
    @test lookup(view, mi) === (ci, results(view, ci))
    results(view, ci).tag = :seeded
    @test results(view, ci).tag === :seeded

    # no generic CodeInstance is inferred along the way
    @test get(cache, mi, nothing) === nothing

    # hits return the published CodeInstance; freshly constructed seeds are egal
    @test typeinf!(interp, mi, Any[CC.Const(seeded_add), CC.Const(1), CC.Const(2)]) === ci
    @test get(seeded(cache, Any[CC.Const(seeded_add), CC.Const(1), CC.Const(2)]), mi,
              nothing) === ci

    # the key is a snapshot: reusing the caller's buffer does not change it
    argtypes[3] = CC.Const(7)
    @test get(seeded(cache, Any[CC.Const(seeded_add), CC.Const(1), CC.Const(2)]), mi,
              nothing) === ci
    @test get(seeded(cache, argtypes), mi, nothing) === nothing

    # other seeds, partially seeded arguments and generic inference are separate
    other = typeinf!(interp, mi, Any[CC.Const(seeded_add), CC.Const(10), CC.Const(20)])
    @test other !== ci && other.rettype_const === 30
    partial_at = Any[CC.Const(seeded_add), Int, CC.Const(2)]
    partial = typeinf!(interp, mi, partial_at)
    @test partial !== ci && partial.rettype === Int
    @test get(seeded(cache, partial_at), mi, nothing) === partial
    generic = typeinf!(interp, mi)
    @test generic !== ci && generic.owner === :SeededTest
    @test get(cache, mi, nothing) === generic
    @test get(view, mi, nothing) === ci

    # the seeded owner is private: it collides with no ordinary owner, and is
    # scoped by the base owner
    tuple_owner = (cache.owner, Tuple(Any[CC.Const(seeded_add), CC.Const(1), CC.Const(2)]))
    @test get(CacheView{SeededResults}(tuple_owner, world), mi, nothing) === nothing
    other_base = CacheView{SeededResults}(:OtherSeededTest, world)
    @test get(seeded(other_base, Any[CC.Const(seeded_add), CC.Const(1), CC.Const(2)]), mi,
              nothing) === nothing

    # several results types share the seeded CodeInstance, each with its own struct
    other_view = seeded(CacheView{TestResults}(cache.owner, world),
                        Any[CC.Const(seeded_add), CC.Const(1), CC.Const(2)])
    @test get(other_view, mi, nothing) === ci
    @test results(other_view, ci) isa TestResults
    @test results(view, ci).tag === :seeded

    # varargs: seeds are given unpacked, as at a call site
    va_mi = method_instance(seeded_va, (Int, Int, Int); world)
    va_at = Any[CC.Const(seeded_va), CC.Const(1), CC.Const(2), CC.Const(3)]
    va_ci = typeinf!(interp, va_mi, va_at)
    @test va_ci isa Core.CodeInstance && va_ci.rettype_const === 6
    @test typeinf!(interp, va_mi, copy(va_at)) === va_ci
    @test get(seeded(cache, va_at), va_mi, nothing) === va_ci

    # the seeded view doesn't depend on the owner type a view declares
    any_cache = CacheView{Any,SeededResults}(cache.owner, world)
    @test get(seeded(any_cache, Any[CC.Const(seeded_add), CC.Const(1), CC.Const(2)]), mi,
              nothing) === ci

    # a MethodInstance widened to a Vararg, seeded with several arities
    wide_mi = CC.specialize_method(only(methods(seeded_va)),
                                   Tuple{typeof(seeded_va), Int, Vararg{Int}}, Core.svec())
    for n in 2:3
        wide_at = Any[CC.Const(seeded_va), (CC.Const(i) for i in 1:n)...]
        wide_ci = typeinf!(interp, wide_mi, wide_at)
        @test wide_ci isa Core.CodeInstance && wide_ci.rettype_const === sum(1:n)
    end

    # type-valued seeds
    conv_mi = method_instance(convert, (Type{Float64}, Int); world)
    conv_at = Any[CC.Const(convert), CC.Const(Float64), Int]
    conv_ci = typeinf!(interp, conv_mi, conv_at)
    @test conv_ci.rettype === Float64
    @test get(seeded(cache, Any[CC.Const(convert), CC.Const(Float64), Int]), conv_mi,
              nothing) === conv_ci
end

# A callee only reachable once the constant is known: with four methods, generic
# inference of `seeded_kernel(::Float32, ::Int)` gives up on `seeded_scale(::Val)`
# and records no edge to the method the seeded source folds in.
seeded_scale(::Val{8}) = 1f0
seeded_scale(::Val{16}) = 1f0
seeded_scale(::Val{32}) = 2f0
seeded_scale(::Val{64}) = 1f0
seeded_kernel(x::Float32, n::Int) = x * seeded_scale(Val(n))
seeded_unrelated(x) = x

# Dispatch to a fallback, which a more specific method inserted later replaces.
seeded_pick(::Val) = 1f0
seeded_insert_kernel(x::Float32, n::Int) = x * seeded_pick(Val(n))

function seeded_kernel_ci(owner, f)
    world = Base.get_world_counter()
    cache = CacheView{SeededResults}(owner, world)
    mi = method_instance(f, (Float32, Int); world)
    ci = typeinf!(SeededInterpreter(cache), mi, Any[CC.Const(f), Float32, CC.Const(32)])
    return ci, float_consts(get_source(ci))
end

@testset "const-seeded invalidation" begin
    ci, consts = Base.invokelatest(seeded_kernel_ci, :SeededInvalidation, seeded_kernel)
    @test consts == [2f0]
    @test ci.max_world == typemax(UInt) broken=!seeds_extended
    @test Base.invokelatest(seeded_kernel_ci, :SeededInvalidation, seeded_kernel)[1] === ci

    # an unrelated world bump keeps the result
    @eval seeded_unrelated(x::Int) = x + 1
    @test Base.invokelatest(seeded_kernel_ci, :SeededInvalidation,
                            seeded_kernel)[1] === ci broken=!seeds_extended
    @test ci.max_world == typemax(UInt) broken=!seeds_extended

    # replacing the callee only seeded inference reaches expires it
    old_world = Base.get_world_counter()
    @eval seeded_scale(::Val{32}) = 3f0
    @test ci.max_world == old_world broken=!seeds_extended
    new_ci, new_consts = Base.invokelatest(seeded_kernel_ci, :SeededInvalidation,
                                           seeded_kernel)
    @test new_ci !== ci
    @test new_consts == [3f0]

    # lookups in the old world still find the old result
    old_mi = method_instance(seeded_kernel, (Float32, Int); world=old_world)
    old_cache = CacheView{SeededResults}(:SeededInvalidation, old_world)
    @test get(seeded(old_cache, Any[CC.Const(seeded_kernel), Float32, CC.Const(32)]),
              old_mi, nothing) === ci broken=!seeds_extended

    # inserting a method that the seeded call now dispatches to expires it too,
    # while one that cannot match does not
    ins_ci, ins_consts = Base.invokelatest(seeded_kernel_ci, :SeededInsertion,
                                           seeded_insert_kernel)
    @test ins_consts == [1f0]
    @eval seeded_pick(::Val{7}) = 7f0
    @test Base.invokelatest(seeded_kernel_ci, :SeededInsertion,
                            seeded_insert_kernel)[1] === ins_ci broken=!seeds_extended
    @eval seeded_pick(::Val{32}) = 2f0
    @test ins_ci.max_world != typemax(UInt)
    ins_ci2, ins_consts2 = Base.invokelatest(seeded_kernel_ci, :SeededInsertion,
                                             seeded_insert_kernel)
    @test ins_ci2 !== ins_ci
    @test ins_consts2 == [2f0]
end

@testset "const-seeded source recovery" begin
    world = Base.get_world_counter()
    cache = CacheView{SeededResults}(:SeededRecovery, world)
    mi = method_instance(seeded_kernel, (Float32, Int); world)
    at = Any[CC.Const(seeded_kernel), Float32, CC.Const(32)]
    view = seeded(cache, at)
    ci = typeinf!(SeededInterpreter(cache), mi, at)
    src = get_source(ci)
    results(view, ci).tag = :old

    # a package image may drop the source: lookups then pass over the CodeInstance,
    # whose results belong to that source, and the seeds are inferred again
    @atomic ci.inferred = nothing
    @test get(view, mi, nothing) === nothing
    @test get_codeinfos(SeededInterpreter(cache), ci) == []
    new_ci = typeinf!(SeededInterpreter(cache), mi, at)
    @test new_ci !== ci
    @test float_consts(get_source(new_ci)) == float_consts(src)
    @test get(view, mi, nothing) === new_ci
    @test results(view, new_ci).tag === nothing
    @test typeinf!(SeededInterpreter(cache), mi, at) === new_ci
    @test results(view, ci).tag === :old

    # a CodeInstance returned for its source in an interpreter's codegen cache gets it
    codegen = IdDict{Core.CodeInstance, Core.CodeInfo}()
    codegen_at = Any[CC.Const(seeded_kernel), Float32, CC.Const(64)]
    codegen_ci = typeinf!(SeededInterpreter(cache; codegen), mi, codegen_at)
    @atomic codegen_ci.inferred = nothing
    codegen_ci = typeinf!(SeededInterpreter(cache; codegen), mi, codegen_at)
    @test get_source(codegen_ci) isa Core.CodeInfo
    @test get(seeded(cache, codegen_at), mi, nothing) === codegen_ci

    # const-return results synthesize their source
    add_mi = method_instance(seeded_add, (Int, Int); world)
    add_at = Any[CC.Const(seeded_add), CC.Const(1), CC.Const(2)]
    add_ci = typeinf!(SeededInterpreter(cache), add_mi, add_at)
    # code coverage taints the effects, so the result is not const-return then
    const_return = VERSION >= v"1.12-" && Base.JLOptions().code_coverage == 0
    @test VERSION < v"1.12-" || CC.use_const_api(add_ci) == const_return
    @atomic add_ci.inferred = nothing
    @test get(seeded(cache, add_at), add_mi, nothing) === (const_return ? add_ci : nothing)
end

# A generated function whose generator fails while `SEEDED_GEN_FAIL` is set:
# inference then has no source to infer.
const SEEDED_GEN_FAIL = Ref(true)
@generated function seeded_generated(x, n)
    SEEDED_GEN_FAIL[] && error("generator failure")
    return :(x * n)
end

@testset "const-seeded inference failure" begin
    world = Base.get_world_counter()
    cache = CacheView{SeededResults}(:SeededFailure, world)
    mi = method_instance(seeded_add, (Int, Int); world)
    at = Any[CC.Const(seeded_add), Int, CC.Const(2)]

    # an exception during inference publishes nothing and leaves nothing reserved
    fail = Ref(true)
    @test_throws ErrorException typeinf!(SeededInterpreter(cache; fail), mi, at)
    @test get(seeded(cache, at), mi, nothing) === nothing
    @test isempty(CompilerCaching.inference_reservations)
    fail[] = false
    ci = typeinf!(SeededInterpreter(cache; fail), mi, at)
    @test ci isa Core.CodeInstance
    @test get(seeded(cache, at), mi, nothing) === ci

    # inference without source returns nothing, then succeeds once there is source
    gen_mi = method_instance(seeded_generated, (Int, Int); world)
    gen_at = Any[CC.Const(seeded_generated), Int, CC.Const(3)]
    @test typeinf!(SeededInterpreter(cache), gen_mi, gen_at) === nothing
    @test get(seeded(cache, gen_at), gen_mi, nothing) === nothing
    @test isempty(CompilerCaching.inference_reservations)
    SEEDED_GEN_FAIL[] = false
    gen_ci = typeinf!(SeededInterpreter(cache), gen_mi, gen_at)
    @test gen_ci isa Core.CodeInstance
    @test gen_ci.rettype === Int
end

# A deep callee chain widens the window in which concurrent first uses race.
for i in 1:32
    @eval @noinline $(Symbol(:seeded_deep_, i))(x, k) =
        $(i == 1 ? :(x * k + 1) : :($(Symbol(:seeded_deep_, i-1))(x, k) + $i))
end
seeded_deep_kernel(x, k) = seeded_deep_32(x, k)

@testset "concurrent const-seeded inference" begin
    ntasks = max(2 * Threads.nthreads(), 8)
    rounds = Threads.nthreads() > 1 ? 10 : 2
    for round in 1:rounds
        world = Base.get_world_counter()
        cache = CacheView{SeededResults}(Symbol("SeededRace", round), world)
        mi = method_instance(seeded_deep_kernel, (Int, Int); world)

        # the same seeds from every task, each with separately constructed owners
        # and interpreters: one CodeInstance, which every task gets
        seen = Vector{Any}(undef, ntasks)
        @sync for t in 1:ntasks
            Threads.@spawn begin
                at = Any[CC.Const(seeded_deep_kernel), Int, CC.Const(32)]
                seen[t] = typeinf!(SeededInterpreter(cache), mi, at)
            end
        end
        view = seeded(cache, Any[CC.Const(seeded_deep_kernel), Int, CC.Const(32)])
        @test all(ci -> ci === seen[1], seen)
        @test get(view, mi, nothing) === seen[1]
        @test length(owned_cis(mi, view.owner)) == 1

        # distinct seeds per task: one CodeInstance each
        @sync for t in 1:ntasks
            Threads.@spawn begin
                at = Any[CC.Const(seeded_deep_kernel), Int, CC.Const(16t)]
                seen[t] = typeinf!(SeededInterpreter(cache), mi, at)
                results(seeded(cache, at), seen[t]).tag = 16t
            end
        end
        @test allunique(seen)
        for t in 1:ntasks
            at = Any[CC.Const(seeded_deep_kernel), Int, CC.Const(16t)]
            @test get(seeded(cache, at), mi, nothing) === seen[t]
            @test results(seeded(cache, at), seen[t]).tag == 16t
        end
        @test get(cache, mi, nothing) === nothing
    end
end
