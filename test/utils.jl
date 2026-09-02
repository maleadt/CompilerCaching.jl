import CompilerCaching: StackedMethodTable
import Core.Compiler: findsup, findall, isoverlayed

# Create method tables for testing
Base.Experimental.@MethodTable(TestMT)
Base.Experimental.@MethodTable(OtherMT)

stacked_ambiguous(x, y) = 0
Base.Experimental.@overlay TestMT stacked_ambiguous(x::Integer, y) = 1
Base.Experimental.@overlay TestMT stacked_ambiguous(x, y::Integer) = 2

# Helper to create different method table views
OverlayMT() = Core.Compiler.OverlayMethodTable(Base.get_world_counter(), TestMT)
StackedMT() = StackedMethodTable(Base.get_world_counter(), TestMT)
DoubleStackedMT() = StackedMethodTable(Base.get_world_counter(), OtherMT, TestMT)

@testset "StackedMethodTable" begin
    @testset "stacks over global MT" begin
        world = Base.get_world_counter()
        stacked = StackedMethodTable(world, TestMT)

        # Should find Base.sin in global MT
        sig = Tuple{typeof(sin), Float64}
        result = findsup(sig, stacked)
        @test result[1] !== nothing
        @test result[1].method.name === :sin
    end

    @testset "isoverlayed" begin
        world = Base.get_world_counter()
        stacked = StackedMethodTable(world, TestMT)
        @test isoverlayed(stacked) == true
    end

    @testset "matches OverlayMethodTable for unoverlayed" begin
        # Without overlay methods, StackedMT should match OverlayMT
        o_sin = findsup(Tuple{typeof(sin), Float64}, OverlayMT())
        s_sin = findsup(Tuple{typeof(sin), Float64}, StackedMT())
        ss_sin = findsup(Tuple{typeof(sin), Float64}, DoubleStackedMT())
        @test s_sin == o_sin
        @test ss_sin == o_sin
    end

    @testset "findall returns merged results" begin
        world = Base.get_world_counter()
        stacked = StackedMethodTable(world, TestMT)

        sig = Tuple{typeof(sin), Number}
        result = findall(sig, stacked; limit=10)
        @test result !== nothing
        @test length(result.matches) >= 1
    end

    @static if VERSION >= v"1.14.0-DEV.3088"
        @testset "findall includes ambiguous methods" begin
            world = Base.get_world_counter()
            stacked = StackedMethodTable(world, TestMT)
            sig = Tuple{typeof(stacked_ambiguous), Integer, Integer}
            result = findall(sig, stacked; limit=10, include_ambiguous=true)
            @test result !== nothing
            @test result.ambig
            @test length(result.matches) == 2
            base_method = only(methods(stacked_ambiguous))
            @test all(match -> match.method !== base_method, result.matches)
        end
    end
end
