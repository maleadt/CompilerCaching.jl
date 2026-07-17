using CompilerCaching
using CompilerCaching: get_source
using Test
using Base.Experimental: @MethodTable

# Common results struct for tests
mutable struct TestResults
    result::Any
    TestResults() = new(nothing)
end

const InfCacheT = @static if isdefined(Core.Compiler, :InferenceCache)
    Core.Compiler.InferenceCache
else
    Vector{Core.Compiler.InferenceResult}
end

@testset "CompilerCaching" verbose=true begin

@testset "basic caching" begin
    # Define a regular Julia function (in global MT)
    global_test_fn(x::Int) = x + 100
    other_fn(x::Int) = x - 1  # used below to test `lookup` miss

    compile_count = Ref(0)
    function my_compile(mi)
        compile_count[] += 1
        # For global methods, source may be compressed (String/Vector{UInt8})
        # Use Base.uncompressed_ast() if you need CodeInfo
        # Here we just return the method name
        mi.def.name
    end

    world = Base.get_world_counter()
    cache = CacheView{TestResults}(:GlobalTest, world)
    mi = method_instance(global_test_fn, (Int,); world)

    # First call - cache miss, compile
    ci = get!(cache, mi) do
        create_ci(cache, mi)
    end
    res = results(cache, ci)
    @test res.result === nothing
    res.result = my_compile(mi)
    @test res.result === :global_test_fn
    @test compile_count[] == 1

    # Cache hit - same CI returned
    ci2 = get!(cache, mi) do
        create_ci(cache, mi)
    end
    res2 = results(cache, ci2)
    @test res2.result === :global_test_fn
    @test ci2 === ci
    @test res2 === res
    @test compile_count[] == 1  # unchanged

    # `lookup` returns the same (ci, res) as separate get + results
    hit = lookup(cache, mi)
    @test hit isa Tuple{Core.CodeInstance, TestResults}
    @test hit[1] === ci
    @test hit[2] === res

    # `lookup` miss when no CI is cached for the MI
    other_mi = method_instance(other_fn, (Int,); world)
    @test lookup(cache, other_mi) === nothing

    # Results structs for different `V`s coexist on the same CI: a cache view
    # parameterized with another V finds the same CI and lazily attaches a fresh
    # results struct alongside the existing one.
    mutable struct OtherResults
        result::Any
        OtherResults() = new(nothing)
    end
    cache_otherV = CacheView{Symbol, OtherResults}(:GlobalTest, world)
    @test get(cache_otherV, mi, nothing) === ci  # same owner → finds the CI
    other_hit = lookup(cache_otherV, mi)
    @test other_hit isa Tuple{Core.CodeInstance, OtherResults}
    @test other_hit[1] === ci
    @test other_hit[2].result === nothing       # fresh, independent results
    @test results(cache, ci) === res            # original results untouched
    @test results(cache_otherV, ci) === other_hit[2]  # stable across accesses
end

@testset "compileable signatures" begin
    # method_instance should return compileable MIs (with vararg widening),
    # not exact-type MIs like method_instances (plural) does.
    world = Base.get_world_counter()
    mi = method_instance(println, (Int, Int, Int, Int); world)
    @test mi !== nothing
    # jl_get_specialization1 is the ground truth for compileable signatures
    sig = Base.signature_type(println, (Int, Int, Int, Int))
    @static if VERSION >= v"1.13-"
        expected = ccall(:jl_get_specialization1, Any, (Any, Csize_t, Cint),
                         sig, world, Cint(0))::Core.MethodInstance
    elseif VERSION >= v"1.12-"
        ptr = ccall(:jl_get_specialization1, Ptr{Cvoid}, (Any, Csize_t, Cint),
                    sig, world, Cint(0))
        expected = unsafe_pointer_to_objref(ptr)::Core.MethodInstance
    else
        min_valid = Ref{Csize_t}(1)
        max_valid = Ref{Csize_t}(typemax(Csize_t))
        ptr = ccall(:jl_get_specialization1, Ptr{Cvoid},
                    (Any, Csize_t, Ref{Csize_t}, Ref{Csize_t}, Cint),
                    sig, world, min_valid, max_valid, Cint(0))
        expected = unsafe_pointer_to_objref(ptr)::Core.MethodInstance
    end
    @test mi === expected
end

@testset "type-valued callees and arguments" begin
    # JuliaLang/julia#62001: closed type-valued callees and arguments dispatch on
    # Core.TypeEgal keys, so `Type{T}` spellings in `tt` need normalization.
    world = Base.get_world_counter()

    # type-valued callee
    mi = method_instance(Vector{Int}, (typeof(undef), Int); world)
    @test mi !== nothing

    # type-valued argument, spelled as Type{T}
    mi = method_instance(identity, (Type{Int},); world)
    @test mi !== nothing
    mi = match_method_instance(identity, (Type{Int},); world)
    @test mi !== nothing
end

@testset "cache partitioning" begin
    # Global MT with sharding keys (e.g., for GPUCompiler-style usage)
    # Different caches for different key combinations
    global_sharded_fn(x::Float64) = x * 2.0

    compile_count = Ref(0)
    function my_compile(mi)
        compile_count[] += 1
        mi.def.name
    end

    world = Base.get_world_counter()
    cache1 = CacheView{TestResults}((:GlobalShardTest, #=opt_level=# 1), world)
    cache2 = CacheView{TestResults}((:GlobalShardTest, #=opt_level=# 2), world)
    mi = method_instance(global_sharded_fn, (Float64,); world)

    # Different sharding keys = different cache entries
    ci1 = get!(cache1, mi) do
        create_ci(cache1, mi)
    end
    res1 = results(cache1, ci1)
    res1.result = my_compile(mi)
    @test res1.result === :global_sharded_fn
    @test compile_count[] == 1

    ci2 = get!(cache2, mi) do
        create_ci(cache2, mi)
    end
    res2 = results(cache2, ci2)
    @test res2.result === nothing  # different shard, not populated
    res2.result = my_compile(mi)
    @test compile_count[] == 2

    # Cache hit on first shard
    ci3 = get!(cache1, mi) do
        create_ci(cache1, mi)
    end
    res3 = results(cache1, ci3)
    @test res3.result === :global_sharded_fn
    @test ci3 === ci1
    @test res3 === res1
    @test compile_count[] == 2  # unchanged
end

@testset "overlay method tables" begin
    # overlay_double is defined at top level with @overlay
    method_table = @eval @MethodTable $(gensym(:method_table))
    overlay_double_name = gensym("overlay_double")
    overlay_double = @eval begin
        function $overlay_double_name end
        Base.Experimental.@overlay $method_table function $overlay_double_name(x::Int)
            x * 2
        end
        $overlay_double_name
    end

    compile_count = Ref(0)
    function my_compile(mi)
        compile_count[] += 1
        # Source is compressed Julia source (can use Base.uncompressed_ast if needed)
        # Return something based on the method
        mi.def.name
    end

    world = Base.get_world_counter()
    cache = CacheView{TestResults}(:OverlayTest, world)
    mi = method_instance(overlay_double, (Int,); world, method_table)

    ci = get!(cache, mi) do
        create_ci(cache, mi)
    end
    res = results(cache, ci)
    res.result = my_compile(mi)
    # Overlay methods may have gensym'd names like "#overlay_double"
    @test occursin("overlay_double", string(res.result))
    @test compile_count[] == 1

    # Cache hit
    ci2 = get!(cache, mi) do
        create_ci(cache, mi)
    end
    res2 = results(cache, ci2)
    @test occursin("overlay_double", string(res2.result))
    @test ci2 === ci
    @test res2 === res
    @test compile_count[] == 1  # unchanged
end

@testset "inference integration" begin
    # Results struct for inference tests
    mutable struct InferenceResults
        ir::Any
        InferenceResults() = new(nothing)
    end

    # Test interpreter that properly integrates with cache
    struct TestInterpreter <: Core.Compiler.AbstractInterpreter
        world::UInt
        cache::CacheView
        inf_cache::InfCacheT
    end
    TestInterpreter(cache::CacheView) =
        TestInterpreter(cache.world, cache, InfCacheT())
    Core.Compiler.cache_owner(interp::TestInterpreter) = interp.cache.owner

    Core.Compiler.InferenceParams(::TestInterpreter) = Core.Compiler.InferenceParams()
    Core.Compiler.OptimizationParams(::TestInterpreter) = Core.Compiler.OptimizationParams()
    Core.Compiler.get_inference_cache(interp::TestInterpreter) = interp.inf_cache
    @static if isdefined(Core.Compiler, :get_inference_world)
        Core.Compiler.get_inference_world(interp::TestInterpreter) = interp.world
    else
        Core.Compiler.get_world_counter(interp::TestInterpreter) = interp.world
    end
    Core.Compiler.lock_mi_inference(::TestInterpreter, ::Core.MethodInstance) = nothing
    Core.Compiler.unlock_mi_inference(::TestInterpreter, ::Core.MethodInstance) = nothing

    test_fn(x::Int) = x + 1
    world = Base.get_world_counter()
    cache = CacheView{InferenceResults}(:InferenceTest, world)
    mi = method_instance(test_fn, (Int,); world)

    interp = TestInterpreter(cache)
    typeinf!(interp, mi)

    # CI is stored in cache, retrieve with get
    ci = get(cache, mi, nothing)
    @test ci isa Core.CodeInstance

    # Get CodeInfo via get_source
    src = get_source(ci)
    @test src isa Core.CodeInfo

    # Test that results accessor works on inferred CI
    res = results(cache, ci)
    @test res isa InferenceResults

    # Test const-return functions get results wrapper
    # These functions return a constant and skip optimization, but finish! should still be called
    const_return_fn(x::Int) = nothing  # Returns constant `nothing`
    world2 = Base.get_world_counter()
    cache2 = CacheView{InferenceResults}(:InferenceTest, world2)
    mi2 = method_instance(const_return_fn, (Int,); world=world2)

    interp2 = TestInterpreter(cache2)
    typeinf!(interp2, mi2)

    ci2 = get(cache2, mi2, nothing)
    @test ci2 isa Core.CodeInstance
    # Verify it's actually a const-return CI (skip under coverage as it disables const-return)
    @test Core.Compiler.use_const_api(ci2) skip=(Base.JLOptions().code_coverage > 0)
    # The key test: results attach lazily even for const-return CIs
    @test results(cache2, ci2) isa InferenceResults
end

@testset "const-prop inference" begin
    # Reuse the InferenceResults and TestInterpreter from the previous testset
    mutable struct ConstPropResults
        ir::Any
        ConstPropResults() = new(nothing)
    end

    struct ConstPropInterpreter <: Core.Compiler.AbstractInterpreter
        world::UInt
        cache::CacheView
        inf_cache::InfCacheT
    end
    ConstPropInterpreter(cache::CacheView) =
        ConstPropInterpreter(cache.world, cache, InfCacheT())
    Core.Compiler.cache_owner(interp::ConstPropInterpreter) = interp.cache.owner

    Core.Compiler.InferenceParams(::ConstPropInterpreter) = Core.Compiler.InferenceParams()
    Core.Compiler.OptimizationParams(::ConstPropInterpreter) = Core.Compiler.OptimizationParams()
    Core.Compiler.get_inference_cache(interp::ConstPropInterpreter) = interp.inf_cache
    @static if isdefined(Core.Compiler, :get_inference_world)
        Core.Compiler.get_inference_world(interp::ConstPropInterpreter) = interp.world
    else
        Core.Compiler.get_world_counter(interp::ConstPropInterpreter) = interp.world
    end
    Core.Compiler.lock_mi_inference(::ConstPropInterpreter, ::Core.MethodInstance) = nothing
    Core.Compiler.unlock_mi_inference(::ConstPropInterpreter, ::Core.MethodInstance) = nothing

    add_fn(a, b) = a + b
    sub_fn(a, b) = a - b  # used below to test `lookup` miss
    world = Base.get_world_counter()
    mi = method_instance(add_fn, (Int, Int); world)

    cache = CacheView{ConstPropResults}(:ConstPropTest, world)
    interp = ConstPropInterpreter(cache)

    # 1. Generic inference
    typeinf!(interp, mi)
    ci = get(cache, mi)
    @test ci.rettype === Int
    @test results(cache, ci) isa ConstPropResults

    # 2. Const-seeded inference (same cache, same interp, same CI)
    const_argtypes = Any[Core.Compiler.Const(add_fn), Core.Compiler.Const(1), Core.Compiler.Const(2)]
    typeinf!(cache, interp, mi, const_argtypes)

    # Results accessible via argtypes
    res = results(cache, ci, const_argtypes)
    @test res isa ConstPropResults

    # Source accessible via argtypes
    src = get_source(ci, const_argtypes)
    @test src isa Core.CodeInfo || src === nothing  # nothing if constabi

    # 3. Generic lookup still works
    @test results(cache, ci) isa ConstPropResults
    @test get_source(ci) isa Core.CodeInfo

    # 4. Cache hit on second call (no error, no duplicate)
    typeinf!(cache, interp, mi, const_argtypes)

    # 5. Different constants → separate entry on same CI
    argtypes2 = Any[Core.Compiler.Const(add_fn), Core.Compiler.Const(10), Core.Compiler.Const(20)]
    typeinf!(cache, interp, mi, argtypes2)
    res2 = results(cache, ci, argtypes2)
    @test res2 isa ConstPropResults
    @test res2 !== res  # different V instance

    # 5a. `lookup` hits and misses for both arities
    hit2 = lookup(cache, mi)
    @test hit2 isa Tuple{Core.CodeInstance, ConstPropResults}
    @test hit2[1] === ci
    @test hit2[2] === results(cache, ci)

    hit_const = lookup(cache, mi, const_argtypes)
    @test hit_const isa Tuple{Core.CodeInstance, ConstPropResults}
    @test hit_const[1] === ci
    @test hit_const[2] === res

    hit_const2 = lookup(cache, mi, argtypes2)
    @test hit_const2[2] === res2  # different argtypes → different V instance

    # Miss on unknown argtypes (CI present, no matching const-prop entry)
    miss_argtypes = Any[Core.Compiler.Const(add_fn), Core.Compiler.Const(99), Core.Compiler.Const(99)]
    @test lookup(cache, mi, miss_argtypes) === nothing

    # Miss on unknown MI (no CI cached at all)
    miss_mi = method_instance(sub_fn, (Int, Int); world)
    @test lookup(cache, miss_mi) === nothing
    @test lookup(cache, miss_mi, const_argtypes) === nothing

    # Freshly-constructed Const wrapper around the same value still hits — the
    # `===` fast path resolves it via egal-on-immutable-struct, no `==` dispatch.
    fresh_argtypes = Any[Core.Compiler.Const(add_fn), Core.Compiler.Const(1), Core.Compiler.Const(2)]
    @test fresh_argtypes !== const_argtypes
    @test lookup(cache, mi, fresh_argtypes) === hit_const

    # 6. Varargs method: invoke stmts list args individually, but on Julia 1.11
    #    matching_cache_argtypes packs them into nargs elements. When more args
    #    are passed than nargs, argtypes must be packed to match.
    va_fn(a, b...) = +(a, b...)
    world_va = Base.get_world_counter()
    cache_va = CacheView{ConstPropResults}(:ConstPropVarargs, world_va)
    mi_va = method_instance(va_fn, (Int, Int, Int); world=world_va)
    interp_va = ConstPropInterpreter(cache_va)
    # Provide unpacked argtypes (4 elements for nargs=3, as an invoke would)
    va_argtypes = Any[Core.Compiler.Const(va_fn), Core.Compiler.Const(1),
                      Core.Compiler.Const(2), Core.Compiler.Const(3)]
    typeinf!(cache_va, interp_va, mi_va, va_argtypes)
end

@static if VERSION >= v"1.12-"
@testset "stale const-prop tombstones" begin
    # A tombstoned (LimitedAccuracy) const-prop result left in the local cache by
    # the root cascade must not suppress const-prop during the callee walk's
    # standalone re-inference (JuliaGPU/CUDA.jl#3185).
    @inline function set_flag!(p::Ptr{Int64}, sym::Symbol, x::Ptr{UInt8})
        if sym === :b
            unsafe_store!(reinterpret(Ptr{Int32}, p), x)     # dead unless const-prop elides it
        elseif sym === :a
            unsafe_store!(reinterpret(Ptr{Ptr{UInt8}}, p), x)
        end
    end
    @noinline function tombstone_thrower(p::Ptr{Int64}, x::Ptr{UInt8})
        set_flag!(p, :a, x)
        throw(nothing)
    end
    function tombstone_kernel(p::Ptr{Int64}, x::Ptr{UInt8}, i::Int)
        i < 1 && tombstone_thrower(p, x)
        unsafe_store!(p, i)
        return
    end

    world = Base.get_world_counter()
    cache = CacheView{TestResults}(:TombstoneTest, world)
    interp = TestInterpreter(cache.world, cache, InfCacheT())

    # seed a tombstone for `set_flag!(::Ptr{Int64}, Const(:a), ::Ptr{UInt8})`
    si_mi = method_instance(set_flag!, (Ptr{Int64}, Symbol, Ptr{UInt8}); world)
    𝕃 = Core.Compiler.typeinf_lattice(interp)
    argtypes = Core.Compiler.matching_cache_argtypes(𝕃, si_mi)
    argtypes[3] = Core.Const(:a)
    overridden = falses(length(argtypes))
    overridden[3] = true
    seed = Core.Compiler.InferenceResult(si_mi, argtypes, overridden)
    seed.tombstone = true
    push!(Core.Compiler.get_inference_cache(interp), seed)

    mi = method_instance(tombstone_kernel, (Ptr{Int64}, Ptr{UInt8}, Int); world)
    @test typeinf!(interp, mi) !== nothing

    th_mi = method_instance(tombstone_thrower, (Ptr{Int64}, Ptr{UInt8}); world)
    th_ci = get(cache, th_mi, nothing)
    @test th_ci !== nothing
    src = get_source(th_ci)
    # with healthy const-prop, the `:b` branch (and its Ptr{Int32} store) is elided
    @test !any(stmt -> occursin("Int32", string(stmt)), src.code)
end

@testset "unresolved invoke targets" begin
    # Inlining's `compileable_specialization` leaves an `:invoke` target as a bare
    # MethodInstance when the compileable specialization is not cached at optimization
    # time; codegen lowers such statements to runtime dispatch. Cached optimized
    # source can therefore reference callees that a plain CodeInstance-edge walk
    # cannot follow (JuliaGPU/Metal.jl checked-conversion overlay failures on 1.12+).
    # `get_codeinfos(interp, ci)` must resolve concrete targets to CodeInstances.
    mod = @eval module $(gensym())
        using Base.Experimental: @MethodTable, @overlay
        @MethodTable method_table
        function overlay_child end
        @overlay method_table @noinline overlay_child(x::UInt32, p::Ptr{Int32}) =
            unsafe_store!(p, x % Int32)

        @noinline child(x::UInt32, p::Ptr{Int32}) = unsafe_store!(p, x % Int32)
        @noinline nospec_child(p::Ptr{Int32}, @nospecialize(x)) =
            unsafe_store!(p, Int32(2))
        function kernel(x::UInt32, p::Ptr{Int32})
            child(x, p)
            nospec_child(p, x)
            return
        end
    end

    get_ci_mi = Core.Compiler.get_ci_mi
    function find_invoke(src::Core.CodeInfo, pred)
        findfirst(eachindex(src.code)) do pc
            stmt = src.code[pc]
            rhs = stmt isa Expr && stmt.head === :(=) ? stmt.args[2] : stmt
            rhs isa Expr && rhs.head === :invoke && pred(rhs.args[1])
        end
    end
    invoke_target(src::Core.CodeInfo, pc::Int) = begin
        stmt = src.code[pc]
        rhs = stmt isa Expr && stmt.head === :(=) ? stmt.args[2] : stmt
        rhs.args[1]
    end

    world = Base.get_world_counter()
    cache = CacheView{TestResults}(:UnresolvedInvokeTest, world)
    interp = TestInterpreter(cache.world, cache, InfCacheT())

    mi = method_instance(mod.kernel, (UInt32, Ptr{Int32}); world)
    root = typeinf!(interp, mi)
    @test root isa Core.CodeInstance

    overlay_mi = method_instance(mod.overlay_child, (UInt32, Ptr{Int32});
                                 world, method_table=mod.method_table)
    @test overlay_mi isa Core.MethodInstance

    # sources that need no rewriting are returned as-is, not copied (only checkable
    # when the stored source is an uncompressed CodeInfo)
    stored = @atomic :monotonic root.inferred
    if stored isa Core.CodeInfo
        @test only(src for (ci, src) in get_codeinfos(interp, root) if ci === root) ===
              stored
    end

    # reconstruct the problematic cache state: retarget the `child` invoke (a
    # CodeInstance edge from fresh inference) to the never-inferred overlay MI
    doctored = copy(get_source(root))
    child_pc = find_invoke(doctored, op -> op isa Core.CodeInstance &&
        get_ci_mi(op).def in methods(mod.child))
    @test child_pc !== nothing
    stmt = doctored.code[child_pc]
    rhs = stmt isa Expr && stmt.head === :(=) ? stmt.args[2] : stmt
    new_rhs = Expr(:invoke, overlay_mi, rhs.args[2:end]...)
    doctored.code[child_pc] = stmt === rhs ? new_rhs : Expr(:(=), stmt.args[1], new_rhs)
    @atomic root.inferred = doctored

    # the walk infers the unresolved callee, includes it, and rewrites the operand
    pairs = get_codeinfos(interp, root)
    overlay_idx = findfirst(p -> get_ci_mi(p.first) === overlay_mi, pairs)
    @test overlay_idx !== nothing
    root_src = only(src for (ci, src) in pairs if ci === root)
    @test invoke_target(root_src, child_pc) === pairs[overlay_idx].first
    # copy-on-write: the cached source still carries the MethodInstance operand
    @test root_src !== doctored
    @test invoke_target(doctored, child_pc) === overlay_mi

    # targets with non-concrete signatures (@nospecialize-widened compileable
    # signatures) keep their runtime-dispatch fallback
    nospec_pc = find_invoke(doctored, op -> begin
        op_mi = op isa Core.CodeInstance ? get_ci_mi(op) : op
        op_mi isa Core.MethodInstance && op_mi.def in methods(mod.nospec_child)
    end)
    @test nospec_pc !== nothing
    if invoke_target(doctored, nospec_pc) isa Core.MethodInstance
        @test invoke_target(root_src, nospec_pc) === invoke_target(doctored, nospec_pc)
        @test !any(p -> get_ci_mi(p.first).def in methods(mod.nospec_child), pairs)
    end

    # const-specialized collection receives the same treatment, seeded from the
    # const-optimized root source
    const_argtypes = Any[Core.Compiler.Const(mod.kernel), Core.Compiler.Const(UInt32(42)),
                         Ptr{Int32}]
    typeinf!(cache, interp, mi, const_argtypes)
    const_src = get_source(root, const_argtypes)
    @test const_src isa Core.CodeInfo
    const_pc = find_invoke(const_src, op -> op isa Core.CodeInstance &&
        get_ci_mi(op).def in methods(mod.child))
    @test const_pc !== nothing
    stmt = const_src.code[const_pc]
    rhs = stmt isa Expr && stmt.head === :(=) ? stmt.args[2] : stmt
    new_rhs = Expr(:invoke, overlay_mi, rhs.args[2:end]...)
    const_src.code[const_pc] = stmt === rhs ? new_rhs : Expr(:(=), stmt.args[1], new_rhs)

    const_pairs = get_codeinfos(interp, root, const_argtypes)
    @test any(p -> get_ci_mi(p.first) === overlay_mi, const_pairs)
    const_root_src = only(src for (ci, src) in const_pairs if ci === root)
    @test invoke_target(const_root_src, const_pc) isa Core.CodeInstance
    @test get_ci_mi(invoke_target(const_root_src, const_pc)) === overlay_mi
    # the stored const-prop entry itself is left unmutated
    @test const_root_src !== const_src
    @test invoke_target(const_src, const_pc) === overlay_mi
end
end

#==============================================================================#
# Custom IR
#==============================================================================#

@testset "custom IR" begin

@testset "basic caching" begin
    method_table = @eval @MethodTable $(gensym(:method_table))

    function basic_node end
    add_method(method_table, basic_node, (Int,), 10)

    compile_count = Ref(0)
    function my_compile(mi)
        source = mi.def.source
        compile_count[] += 1
        source * 2
    end

    world = Base.get_world_counter()
    cache = CacheView{TestResults}(:BasicTest, world)
    mi = method_instance(basic_node, (Int,); world, method_table)

    # First call: cache miss, compile_fn invoked
    ci = get!(cache, mi) do
        create_ci(cache, mi)
    end
    res = results(cache, ci)
    res.result = my_compile(mi)
    @test res.result == 20  # 10 * 2
    @test compile_count[] == 1

    # Second call: cache hit, compile_fn NOT invoked
    ci2 = get!(cache, mi) do
        create_ci(cache, mi)
    end
    res2 = results(cache, ci2)
    @test res2.result == 20
    @test ci2 === ci
    @test res2 === res
    @test compile_count[] == 1  # still 1

    # Redefine method → invalidates cache, recompile
    add_method(method_table, basic_node, (Int,), 30)
    world = Base.get_world_counter()
    cache = CacheView{TestResults}(:BasicTest, world)
    mi = method_instance(basic_node, (Int,); world, method_table)
    ci3 = get!(cache, mi) do
        create_ci(cache, mi)
    end
    res3 = results(cache, ci3)
    res3.result = my_compile(mi)
    @test res3.result == 60  # 30 * 2
    @test compile_count[] == 2  # incremented
end

@testset "multiple dispatch" begin
    method_table = @eval @MethodTable $(gensym(:method_table))

    function dispatch_node end
    add_method(method_table, dispatch_node, (Int,), 100)
    add_method(method_table, dispatch_node, (Float64,), 200)

    compile_count = Ref(0)
    function my_compile(mi)
        source = mi.def.source
        compile_count[] += 1
        source + 1
    end

    world = Base.get_world_counter()
    cache = CacheView{TestResults}(:DispatchTest, world)
    mi_int = method_instance(dispatch_node, (Int,); world, method_table)
    mi_float = method_instance(dispatch_node, (Float64,); world, method_table)

    # Different types → different cache entries, each compiles once
    ci_int = get!(cache, mi_int) do
        create_ci(cache, mi_int)
    end
    res_int = results(cache, ci_int)
    res_int.result = my_compile(mi_int)
    @test res_int.result == 101
    @test compile_count[] == 1

    ci_float = get!(cache, mi_float) do
        create_ci(cache, mi_float)
    end
    res_float = results(cache, ci_float)
    res_float.result = my_compile(mi_float)
    @test res_float.result == 201
    @test compile_count[] == 2

    # Cache hits - no recompilation
    ci_int2 = get!(cache, mi_int) do
        create_ci(cache, mi_int)
    end
    res_int2 = results(cache, ci_int2)
    @test res_int2.result == 101
    @test ci_int2 === ci_int
    @test res_int2 === res_int
    @test compile_count[] == 2  # unchanged

    ci_float2 = get!(cache, mi_float) do
        create_ci(cache, mi_float)
    end
    res_float2 = results(cache, ci_float2)
    @test res_float2.result == 201
    @test ci_float2 === ci_float
    @test res_float2 === res_float
    @test compile_count[] == 2  # unchanged

    # Redefine only Int method → only Int recompiles
    add_method(method_table, dispatch_node, (Int,), 50)
    world = Base.get_world_counter()
    cache = CacheView{TestResults}(:DispatchTest, world)
    mi_int = method_instance(dispatch_node, (Int,); world, method_table)
    ci_int3 = get!(cache, mi_int) do
        create_ci(cache, mi_int)
    end
    res_int3 = results(cache, ci_int3)
    res_int3.result = my_compile(mi_int)
    @test res_int3.result == 51
    @test compile_count[] == 3

    # Float64 still uses cached version (need to re-lookup mi after world change)
    mi_float = method_instance(dispatch_node, (Float64,); world, method_table)
    ci_float3 = get!(cache, mi_float) do
        create_ci(cache, mi_float)
    end
    res_float3 = results(cache, ci_float3)
    @test res_float3.result == 201
    @test compile_count[] == 3  # unchanged
end

@testset "Function-subtype specialization" begin
    # jl_compilation_sig() despecializes Function-subtype arguments to Function when
    # the method's `called` bitmask says the arg is not used in call position.
    # For foreign methods (no Julia source to scan), add_method must preserve the
    # conservative default (0xff) so that specialization is not lost.
    method_table = @eval @MethodTable $(gensym(:method_table))

    struct MyCallable <: Function end
    @test MyCallable <: Function
    @test isconcretetype(MyCallable)

    function callable_node end
    add_method(method_table, callable_node, (Any, Int), :callable_ir)

    world = Base.get_world_counter()
    mi = method_instance(callable_node, (MyCallable, Int); world, method_table)
    @test mi !== nothing
    # specTypes must preserve the concrete MyCallable, not widen to Function
    @test mi.specTypes === Tuple{typeof(callable_node), MyCallable, Int}
end

@testset "missing method" begin
    method_table = @eval @MethodTable $(gensym(:method_table))

    function missing_node end
    # No method registered

    # Returns nothing when no method found
    world = Base.get_world_counter()
    mi = method_instance(missing_node, (Int,); world, method_table)
    @test mi === nothing
end

@testset "dependency invalidation" begin
    method_table = @eval @MethodTable $(gensym(:method_table))

    function parent_node end
    function child_node end
    add_method(method_table, child_node, (Int,), :child_ir)
    add_method(method_table, parent_node, (Int,), :parent_ir)

    child_compile_count = Ref(0)
    parent_compile_count = Ref(0)

    # Child compilation: creates CI with no deps
    function compile_child(cache, mi)
        child_compile_count[] += 1
        ci = get!(cache, mi) do
            create_ci(cache, mi)
        end
        res = results(cache, ci)
        res.result = :child_ir
    end

    # Parent compilation: creates CI with dependency on child
    function compile_parent(cache, mi)
        parent_compile_count[] += 1
        child_mi = method_instance(child_node, (Int,); world=cache.world, method_table)

        # Must compile child first to establish dependency
        compile_child(cache, child_mi)

        # Create CI with dependency
        ci = create_ci(cache, mi; deps=[child_mi])
        cache[mi] = ci
        res = results(cache, ci)
        res.result = :parent_ir
    end

    world = Base.get_world_counter()
    cache = CacheView{TestResults}(:DepTest, world)
    child_mi = method_instance(child_node, (Int,); world, method_table)
    parent_mi = method_instance(parent_node, (Int,); world, method_table)

    # Compile child first
    compile_child(cache, child_mi)
    @test child_compile_count[] == 1

    # Compile parent (depends on child)
    compile_parent(cache, parent_mi)
    @test parent_compile_count[] == 1
    @test child_compile_count[] == 2  # child recompiled during parent compilation

    # Cache hits
    ci_child = get!(cache, child_mi) do
        create_ci(cache, child_mi)
    end
    res_child = results(cache, ci_child)
    @test res_child.result === :child_ir
    ci_parent = get!(cache, parent_mi) do
        create_ci(cache, parent_mi)
    end
    res_parent = results(cache, ci_parent)
    @test res_parent.result === :parent_ir

    # Redefine child → child recompiles
    add_method(method_table, child_node, (Int,), :new_child_ir)
    world = Base.get_world_counter()
    cache = CacheView{TestResults}(:DepTest, world)
    child_mi = method_instance(child_node, (Int,); world, method_table)
    compile_child(cache, child_mi)
    @test child_compile_count[] == 3

    # Parent should also recompile due to dependency (cache miss)
    parent_mi = method_instance(parent_node, (Int,); world, method_table)
    compile_parent(cache, parent_mi)
    @test parent_compile_count[] == 2
end

@testset "binding edges" begin
    # create_ci consults captured_globals(source) to pick up the
    # GlobalRefs a foreign IR captures, so they participate in invalidation.
    # The mechanism exists on 1.12+; on 1.11 it's a no-op.

    binding_mod = Module()
    Core.eval(binding_mod, :(const trait_const = 1))
    gr = GlobalRef(binding_mod, :trait_const)

    # Custom IR carrying its captured GlobalRefs; override the hook for it.
    struct TraitIR
        grefs::Vector{GlobalRef}
    end
    CompilerCaching.captured_globals(ir::TraitIR) = ir.grefs

    method_table = @eval @MethodTable $(gensym(:method_table))
    function trait_node end
    m = add_method(method_table, trait_node, (Int,), TraitIR([gr]))

    @static if VERSION >= v"1.12-"
        # Foreign source: did_scan_source is set so Julia's CodeInfo-only
        # scan never tries to uncompress our IR.
        @test (m.did_scan_source & 0x1) != 0x0

        # create_ci registers the captured bindings as CI-level edges.
        world = Base.get_world_counter()
        cache = CacheView{TestResults}(:TraitBindingTest, world)
        mi = method_instance(trait_node, (Int,); world, method_table)
        ci = get!(cache, mi) do
            create_ci(cache, mi)
        end
        b = convert(Core.Binding, gr)
        @test isdefined(ci, :edges)
        @test any(i -> isassigned(ci.edges, i) && ci.edges[i] === b,
                  eachindex(ci.edges))

        # The CI is also a direct edge of the binding, so invalidation
        # reaches it without going through method-level scanning.
        @test isdefined(b, :backedges)
        @test any(i -> isassigned(b.backedges, i) && b.backedges[i] === ci,
                  eachindex(b.backedges))

        # End-to-end: replacing the const should drive ci.max_world down.
        @test ci.max_world == typemax(UInt)
        Core.eval(binding_mod, :(const trait_const = 2))
        @test ci.max_world < typemax(UInt)
    end

    # CodeInfo source: must NOT pre-set did_scan_source, so Julia's own
    # scan still runs to discover GlobalRefs in the lowered IR.
    @static if VERSION >= v"1.12-"
        plain_method_table = @eval @MethodTable $(gensym(:method_table))
        function plain_node end
        ci_obj = code_lowered(identity, Tuple{Int})[1]
        plain_method = add_method(plain_method_table, plain_node, (Int,), ci_obj)
        @test (plain_method.did_scan_source & 0x1) == 0x0
    end
end

@testset "method table isolation" begin
    method_table_a = @eval @MethodTable $(gensym(:method_table_a))
    method_table_b = @eval @MethodTable $(gensym(:method_table_b))

    function isolated_node end
    add_method(method_table_a, isolated_node, (Int,), :ir_a)
    add_method(method_table_b, isolated_node, (Int,), :ir_b)

    world = Base.get_world_counter()
    cache_a = CacheView{TestResults}(:IsolationA, world)
    cache_b = CacheView{TestResults}(:IsolationB, world)
    mi_a = method_instance(isolated_node, (Int,); world, method_table=method_table_a)
    mi_b = method_instance(isolated_node, (Int,); world, method_table=method_table_b)

    ci_a = get!(cache_a, mi_a) do
        create_ci(cache_a, mi_a)
    end
    res_a = results(cache_a, ci_a)
    res_a.result = mi_a.def.source
    ci_b = get!(cache_b, mi_b) do
        create_ci(cache_b, mi_b)
    end
    res_b = results(cache_b, ci_b)
    res_b.result = mi_b.def.source

    @test res_a.result === :ir_a
    @test res_b.result === :ir_b
end

end

#==============================================================================#
# Examples
#==============================================================================#

@testset "Examples" begin
    function find_sources(path::String, sources=String[])
        if isdir(path)
            for entry in readdir(path)
                find_sources(joinpath(path, entry), sources)
            end
        elseif endswith(path, ".jl")
            push!(sources, path)
        end
        sources
    end

    examples_dir = joinpath(@__DIR__, "..", "examples")
    examples = find_sources(examples_dir)
    filter!(file -> readline(file) != "# EXCLUDE FROM TESTING", examples)

    for example in examples
        name = splitext(basename(example))[1]
        @testset "$name" begin
            cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) $example`
            buf = IOBuffer()
            ok = success(pipeline(cmd; stdout=buf, stderr=buf))
            if !ok
                print(String(take!(buf)))
            end
            @test ok
        end
    end
end

include("utils.jl")
include("precompile.jl")

end
