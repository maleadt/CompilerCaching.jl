using CompilerCaching
using Test
using Base.Experimental: @MethodTable

include("utils.jl")

# Test helper: wraps simple compile functions in three-phase API
function simple_cached_compilation(compile_fn, cache::CompilerCache, mi, world)
    cached_compilation(cache, mi, world;
        infer = (c, m, w) -> begin
            result = compile_fn(m)
            ci = CompilerCaching.cache!(c, m; world=w)
            [ci => result]
        end,
        codegen = (c, m, w, codeinfos) -> only(codeinfos)[2],
        link = (c, m, w, r) -> r
    )
end

@testset "CompilerCaching" verbose=true begin

#==============================================================================#
# Mode 1: Custom IR - Empty methods with add_method
#==============================================================================#

@testset "Mode 1: Custom IR" begin

@testset "basic caching" begin
    method_table = @eval @MethodTable $(gensym(:method_table))
    cache = CompilerCache(:BasicTest)

    function basic_node end
    add_method(method_table, basic_node, (Int,), 10)

    compile_count = Ref(0)
    function my_compile(mi)
        source = mi.def.source
        compile_count[] += 1
        source * 2
    end

    world = Base.get_world_counter()
    mi = method_instance(basic_node, (Int,); world, method_table)

    # First call: cache miss, compile_fn invoked
    r1 = simple_cached_compilation(my_compile, cache, mi, world)
    @test r1 == 20  # 10 * 2
    @test compile_count[] == 1

    # Second call: cache hit, compile_fn NOT invoked
    r2 = simple_cached_compilation(my_compile, cache, mi, world)
    @test r2 == 20
    @test compile_count[] == 1  # still 1

    # Redefine method → invalidates cache, recompile
    add_method(method_table, basic_node, (Int,), 30)
    world = Base.get_world_counter()
    mi = method_instance(basic_node, (Int,); world, method_table)
    r3 = simple_cached_compilation(my_compile, cache, mi, world)
    @test r3 == 60  # 30 * 2
    @test compile_count[] == 2  # incremented
end

@testset "multiple dispatch" begin
    method_table = @eval @MethodTable $(gensym(:method_table))
    cache = CompilerCache(:DispatchTest)

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
    mi_int = method_instance(dispatch_node, (Int,); world, method_table)
    mi_float = method_instance(dispatch_node, (Float64,); world, method_table)

    # Different types → different cache entries, each compiles once
    r_int = simple_cached_compilation(my_compile, cache, mi_int, world)
    @test r_int == 101
    @test compile_count[] == 1

    r_float = simple_cached_compilation(my_compile, cache, mi_float, world)
    @test r_float == 201
    @test compile_count[] == 2

    # Cache hits - no recompilation
    r_int2 = simple_cached_compilation(my_compile, cache, mi_int, world)
    @test r_int2 == 101
    @test compile_count[] == 2  # unchanged

    r_float2 = simple_cached_compilation(my_compile, cache, mi_float, world)
    @test r_float2 == 201
    @test compile_count[] == 2  # unchanged

    # Redefine only Int method → only Int recompiles
    add_method(method_table, dispatch_node, (Int,), 50)
    world = Base.get_world_counter()
    mi_int = method_instance(dispatch_node, (Int,); world, method_table)
    r_int3 = simple_cached_compilation(my_compile, cache, mi_int, world)
    @test r_int3 == 51
    @test compile_count[] == 3

    # Float64 still uses cached version (need to re-lookup mi after world change)
    mi_float = method_instance(dispatch_node, (Float64,); world, method_table)
    r_float3 = simple_cached_compilation(my_compile, cache, mi_float, world)
    @test r_float3 == 201
    @test compile_count[] == 3  # unchanged
end

@testset "nothing on missing method" begin
    method_table = @eval @MethodTable $(gensym(:method_table))

    function missing_node end
    # No method registered

    # Returns nothing when no method found
    world = Base.get_world_counter()
    mi = method_instance(missing_node, (Int,); world, method_table)
    @test mi === nothing
end

@testset "sharding keys" begin
    # Compiler with NamedTuple keys - now keys are stored in the cache
    ShardKeys = @NamedTuple{opt_level::Int, debug::Bool}
    method_table = @eval @MethodTable $(gensym(:method_table))

    # Different caches for different key combinations
    keys1 = (opt_level=1, debug=false)
    keys2 = (opt_level=2, debug=false)
    keys3 = (opt_level=1, debug=true)
    cache1 = CompilerCache{ShardKeys}(:ShardingTest, keys1)
    cache2 = CompilerCache{ShardKeys}(:ShardingTest, keys2)
    cache3 = CompilerCache{ShardKeys}(:ShardingTest, keys3)

    function sharded_node end
    add_method(method_table, sharded_node, (Int,), 42)  # only need to add once, all caches share MT

    compile_count = Ref(0)
    function my_compile(mi)
        compile_count[] += 1
        mi.def.source
    end

    world = Base.get_world_counter()
    mi = method_instance(sharded_node, (Int,); world, method_table)

    # First key combination
    r1 = simple_cached_compilation(my_compile, cache1, mi, world)
    @test r1 == 42
    @test compile_count[] == 1

    # Same key combination → cache hit
    r2 = simple_cached_compilation(my_compile, cache1, mi, world)
    @test r2 == 42
    @test compile_count[] == 1  # unchanged

    # Different key combination → cache miss (different shard)
    r3 = simple_cached_compilation(my_compile, cache2, mi, world)
    @test r3 == 42
    @test compile_count[] == 2  # new compilation

    # Yet another key combination
    r4 = simple_cached_compilation(my_compile, cache3, mi, world)
    @test r4 == 42
    @test compile_count[] == 3  # another new compilation

    # Back to first key combination → still cached
    r5 = simple_cached_compilation(my_compile, cache1, mi, world)
    @test r5 == 42
    @test compile_count[] == 3  # unchanged
end

@testset "three-phase API access to source" begin
    method_table = @eval @MethodTable $(gensym(:method_table))
    cache = CompilerCache(:SourceTest)

    function source_node end
    add_method(method_table, source_node, (Int,), :my_source)

    captured_source = Ref{Any}(nothing)

    world = Base.get_world_counter()
    mi = method_instance(source_node, (Int,); world, method_table)

    function my_compile(m)
        captured_source[] = m.def.source
        :compiled
    end

    result = simple_cached_compilation(my_compile, cache, mi, world)

    @test captured_source[] === :my_source
    @test mi isa Core.MethodInstance
    @test mi.def.name === :source_node
    @test result === :compiled
end

@testset "dependency invalidation" begin
    method_table = @eval @MethodTable $(gensym(:method_table))
    cache = CompilerCache(:DepTest)

    function parent_node end
    function child_node end
    add_method(method_table, child_node, (Int,), :child_ir)
    add_method(method_table, parent_node, (Int,), :parent_ir)

    child_compile_count = Ref(0)
    parent_compile_count = Ref(0)

    # Child infer: creates CI with no deps
    function child_infer(c, m, w)
        child_compile_count[] += 1
        ci = cache!(c, m; world=w)
        [ci => :child_compiled]
    end

    # Parent infer: creates CI with dependency on child
    function parent_infer(c, m, w)
        parent_compile_count[] += 1
        child_mi = method_instance(child_node, (Int,); world=w, method_table)
        ci = cache!(c, m; world=w, deps=[child_mi])
        [ci => :parent_compiled]
    end

    passthrough_codegen(c, m, w, codeinfos) = only(codeinfos)[2]
    passthrough_link(c, m, w, r) = r

    world = Base.get_world_counter()
    child_mi = method_instance(child_node, (Int,); world, method_table)
    parent_mi = method_instance(parent_node, (Int,); world, method_table)

    # Compile child first
    cached_compilation(cache, child_mi, world;
        infer = child_infer, codegen = passthrough_codegen, link = passthrough_link)
    @test child_compile_count[] == 1

    # Compile parent (depends on child)
    cached_compilation(cache, parent_mi, world;
        infer = parent_infer, codegen = passthrough_codegen, link = passthrough_link)
    @test parent_compile_count[] == 1

    # Cache hits
    cached_compilation(cache, child_mi, world;
        infer = child_infer, codegen = passthrough_codegen, link = passthrough_link)
    cached_compilation(cache, parent_mi, world;
        infer = parent_infer, codegen = passthrough_codegen, link = passthrough_link)
    @test child_compile_count[] == 1
    @test parent_compile_count[] == 1

    # Redefine child → child recompiles
    add_method(method_table, child_node, (Int,), :new_child_ir)
    world = Base.get_world_counter()
    child_mi = method_instance(child_node, (Int,); world, method_table)
    cached_compilation(cache, child_mi, world;
        infer = child_infer, codegen = passthrough_codegen, link = passthrough_link)
    @test child_compile_count[] == 2

    # Parent should also recompile due to dependency
    parent_mi = method_instance(parent_node, (Int,); world, method_table)
    cached_compilation(cache, parent_mi, world;
        infer = parent_infer, codegen = passthrough_codegen, link = passthrough_link)
    @test parent_compile_count[] == 2
end

@testset "method table isolation" begin
    method_table_a = @eval @MethodTable $(gensym(:method_table_a))
    cache_a = CompilerCache(:IsolationA)
    method_table_b = @eval @MethodTable $(gensym(:method_table_b))
    cache_b = CompilerCache(:IsolationB)

    function isolated_node end
    add_method(method_table_a, isolated_node, (Int,), :ir_a)
    add_method(method_table_b, isolated_node, (Int,), :ir_b)

    result_a = Ref{Any}(nothing)
    result_b = Ref{Any}(nothing)

    world = Base.get_world_counter()
    mi_a = method_instance(isolated_node, (Int,); world, method_table=method_table_a)
    mi_b = method_instance(isolated_node, (Int,); world, method_table=method_table_b)

    function compile_a(m)
        result_a[] = m.def.source
        m.def.source
    end
    function compile_b(m)
        result_b[] = m.def.source
        m.def.source
    end

    simple_cached_compilation(compile_a, cache_a, mi_a, world)
    simple_cached_compilation(compile_b, cache_b, mi_b, world)

    @test result_a[] === :ir_a
    @test result_b[] === :ir_b
end

@testset "complex source types" begin
    method_table = @eval @MethodTable $(gensym(:method_table))
    cache = CompilerCache(:ComplexTest)

    # Store a complex struct as source
    struct MyIR
        nodes::Vector{Symbol}
        edges::Dict{Symbol, Vector{Symbol}}
    end

    function complex_node end
    ir = MyIR([:a, :b, :c], Dict(:a => [:b], :b => [:c]))
    add_method(method_table, complex_node, (Int,), ir)

    captured_ir = Ref{Any}(nothing)

    world = Base.get_world_counter()
    mi = method_instance(complex_node, (Int,); world, method_table)

    function my_compile(m)
        source = m.def.source
        captured_ir[] = source
        length(source.nodes)
    end

    result = simple_cached_compilation(my_compile, cache, mi, world)
    @test result == 3
    @test captured_ir[] isa MyIR
    @test captured_ir[].nodes == [:a, :b, :c]
    @test captured_ir[].edges[:a] == [:b]
end

end # Mode 1

#==============================================================================#
# Mode 2: Overlay Methods - Julia source in custom MT
#==============================================================================#

@testset "Mode 2: Overlay Methods" begin

@testset "Julia method in custom MT" begin
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
    cache = CompilerCache(:OverlayTest)

    compile_count = Ref(0)
    function my_compile(mi)
        compile_count[] += 1
        # Source is compressed Julia source (can use Base.uncompressed_ast if needed)
        # Return something based on the method
        mi.def.name
    end

    world = Base.get_world_counter()
    mi = method_instance(overlay_double, (Int,); world, method_table)

    result = simple_cached_compilation(my_compile, cache, mi, world)
    # Overlay methods may have gensym'd names like "#overlay_double"
    @test occursin("overlay_double", string(result))
    @test compile_count[] == 1

    # Cache hit
    result2 = simple_cached_compilation(my_compile, cache, mi, world)
    @test occursin("overlay_double", string(result2))
    @test compile_count[] == 1  # unchanged
end

end # Mode 2

#==============================================================================#
# Mode 3: Global Methods - Julia source in global MT
#==============================================================================#

@testset "Mode 3: Global Methods" begin

@testset "global MT lookup" begin
    # Use Compiler without method table = global MT
    cache = CompilerCache(:GlobalTest)

    # Define a regular Julia function (in global MT)
    global_test_fn(x::Int) = x + 100

    compile_count = Ref(0)
    function my_compile(mi)
        compile_count[] += 1
        # For global methods, source may be compressed (String/Vector{UInt8})
        # Use Base.uncompressed_ast() if you need CodeInfo
        # Here we just return the method name
        mi.def.name
    end

    world = Base.get_world_counter()
    mi = method_instance(global_test_fn, (Int,); world)

    result = simple_cached_compilation(my_compile, cache, mi, world)
    @test result === :global_test_fn
    @test compile_count[] == 1

    # Cache hit
    result2 = simple_cached_compilation(my_compile, cache, mi, world)
    @test result2 === :global_test_fn
    @test compile_count[] == 1  # unchanged
end

@testset "global MT with sharding keys" begin
    # Global MT with sharding keys (e.g., for GPUCompiler-style usage)
    # Different caches for different key combinations
    ShardKeys = @NamedTuple{opt_level::Int}
    cache1 = CompilerCache{ShardKeys}(:GlobalShardTest, (opt_level=1,))
    cache2 = CompilerCache{ShardKeys}(:GlobalShardTest, (opt_level=2,))

    global_sharded_fn(x::Float64) = x * 2.0

    compile_count = Ref(0)
    function my_compile(mi)
        compile_count[] += 1
        mi.def.name
    end

    world = Base.get_world_counter()
    mi = method_instance(global_sharded_fn, (Float64,); world)

    # Different sharding keys = different cache entries
    r1 = simple_cached_compilation(my_compile, cache1, mi, world)
    @test r1 === :global_sharded_fn
    @test compile_count[] == 1

    r2 = simple_cached_compilation(my_compile, cache2, mi, world)
    @test r2 === :global_sharded_fn
    @test compile_count[] == 2  # different shard

    r3 = simple_cached_compilation(my_compile, cache1, mi, world)
    @test r3 === :global_sharded_fn
    @test compile_count[] == 2  # cache hit
end

end # Mode 3

#==============================================================================#
# populate!
#==============================================================================#

@testset "populate!" begin
    @testset "basic inference" begin
        cache = CompilerCache(:InferenceTest)

        # Test interpreter that properly integrates with cache
        struct TestInterpreter <: Core.Compiler.AbstractInterpreter
            world::UInt
            cache::CompilerCache
            inf_cache::Vector{Core.Compiler.InferenceResult}
        end
        TestInterpreter(cache::CompilerCache, world::UInt) =
            TestInterpreter(world, cache, Core.Compiler.InferenceResult[])

        Core.Compiler.InferenceParams(::TestInterpreter) = Core.Compiler.InferenceParams()
        Core.Compiler.OptimizationParams(::TestInterpreter) = Core.Compiler.OptimizationParams()
        Core.Compiler.get_inference_cache(interp::TestInterpreter) = interp.inf_cache
        @static if isdefined(Core.Compiler, :get_inference_world)
            Core.Compiler.get_inference_world(interp::TestInterpreter) = interp.world
        else
            Core.Compiler.get_world_counter(interp::TestInterpreter) = interp.world
        end
        Core.Compiler.cache_owner(interp::TestInterpreter) = cache_owner(interp.cache)
        Core.Compiler.lock_mi_inference(::TestInterpreter, ::Core.MethodInstance) = nothing
        Core.Compiler.unlock_mi_inference(::TestInterpreter, ::Core.MethodInstance) = nothing

        test_fn(x::Int) = x + 1
        world = Base.get_world_counter()
        mi = method_instance(test_fn, (Int,); world)

        interp = TestInterpreter(cache, world)
        result = populate!(cache, interp, mi)

        # populate! now always returns Vector{Pair{CI, IR}}
        @test result isa Vector
        @test length(result) >= 1
        ci, ir = first(result)
        @test ci isa Core.CodeInstance
        @static if VERSION >= v"1.12.0-DEV.15"
            @test ir isa Core.CodeInfo
        else
            @test ir === nothing  # 1.11 uses callback-based codegen
        end
    end
end

#==============================================================================#
# disk cache
#==============================================================================#

@static if VERSION >= v"1.12-"
@testset "disk cache" begin
    # Test disk cache by running in subprocess (fresh Julia = no in-memory CI)
    # First process populates the disk cache, second process hits it

    disk_cache_script = """
    using CompilerCaching
    const CC = Core.Compiler

    struct DiskCacheInterpreter <: CC.AbstractInterpreter
        world::UInt
        cache::CompilerCache
        inf_cache::Vector{CC.InferenceResult}
    end
    DiskCacheInterpreter(cache::CompilerCache, world::UInt) =
        DiskCacheInterpreter(world, cache, CC.InferenceResult[])

    CC.InferenceParams(::DiskCacheInterpreter) = CC.InferenceParams()
    CC.OptimizationParams(::DiskCacheInterpreter) = CC.OptimizationParams()
    CC.get_inference_cache(interp::DiskCacheInterpreter) = interp.inf_cache
    CC.get_inference_world(interp::DiskCacheInterpreter) = interp.world
    CC.lock_mi_inference(::DiskCacheInterpreter, ::Core.MethodInstance) = nothing
    CC.unlock_mi_inference(::DiskCacheInterpreter, ::Core.MethodInstance) = nothing
    @setup_caching DiskCacheInterpreter.cache

    infer_count = Ref(0)
    codegen_count = Ref(0)
    link_count = Ref(0)

    function disk_infer(cache, mi, world)
        infer_count[] += 1
        interp = DiskCacheInterpreter(cache, world)
        populate!(cache, interp, mi)
    end
    disk_codegen(cache, mi, world, codeinfos) = (codegen_count[] += 1; :codegen_result)
    disk_link(cache, mi, world, ir_data) = (link_count[] += 1; ir_data)

    world = Base.get_world_counter()
    mi = method_instance(identity, (Int,); world)
    cache = CompilerCache(:DiskCacheTest; disk_cache=true)

    mode = ARGS[1]
    if mode == "populate"
        clear_disk_cache!(cache)
        result = cached_compilation(cache, mi, world;
            infer=disk_infer, codegen=disk_codegen, link=disk_link)
        @assert result === :codegen_result
        @assert infer_count[] == 1
        @assert codegen_count[] == 1
        @assert link_count[] == 1
    elseif mode == "hit"
        result = cached_compilation(cache, mi, world;
            infer=disk_infer, codegen=disk_codegen, link=disk_link)
        @assert result === :codegen_result
        @assert infer_count[] == 1    "infer should be called (fresh CI needs wrapper)"
        @assert codegen_count[] == 0  "codegen should be SKIPPED (disk cache hit)"
        @assert link_count[] == 1     "link should be called"
    elseif mode == "cleanup"
        clear_disk_cache!(cache)
    end
    """

    prj = Base.active_project()
    run_script(mode) = run(`$(Base.julia_cmd()) --project=$prj -e $disk_cache_script -- $mode`)

    @test success(run_script("populate"))
    @test success(run_script("hit"))
    run_script("cleanup")
end
end # @static if

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
            @test success(pipeline(cmd; stdout=devnull, stderr=devnull))
        end
    end
end

end # @testset "CompilerCaching"

println("All tests passed!")
