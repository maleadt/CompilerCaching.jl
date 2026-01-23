# Foreign IR example - bypasses Julia's inference/compiler stack entirely
# - Methods registered with add_method storing custom IR
# - Cache views created on-the-fly before compilation
# - Demonstrates: caching, redefinition, multiple dispatch

using CompilerCaching: CacheView, add_method, method_instance, create_ci, results


Base.Experimental.@MethodTable method_table


## Results struct for foreign compilation

mutable struct ForeignResults
    ir::Any   # The evaluated IR result
    ForeignResults() = new(nothing)
end


## Simple IR: numbers, arithmetic ops as heads, calls to foreign functions
#
# Examples:
#   42                          → literal
#   Expr(:*, 21, 2)             → 21 * 2
#   Expr(:+, 1, Expr(:*, 2, 3)) → 1 + (2 * 3)
#   Expr(:call, myfunc)         → call myfunc()

function interpret(cache, expr, deps)
    # Literals
    expr isa Number && return expr

    # Expressions
    if expr isa Expr
        head, args... = expr.head, expr.args...

        # Arithmetic operations
        if head === :+
            return sum(interpret(cache, a, deps) for a in args)
        elseif head === :*
            return prod(interpret(cache, a, deps) for a in args)
        elseif head === :^
            base, exp = args
            return interpret(cache, base, deps) ^ interpret(cache, exp, deps)
        end

        # Call to function in our method table. This triggers recursive IR generation.
        if head === :call
            f = only(args)::Function
            mi = @something(method_instance(f, (); world=cache.world, method_table),
                            error("Unknown function: $f"))
            ir = compile!(cache, mi)
            push!(deps, mi)
            return ir
        end

        error("Unknown expression head: $head")
    end

    error("Unsupported IR: $expr")
end

const compilations = Ref(0)

function compile!(cache::CacheView, mi::Core.MethodInstance)
    ci = get!(cache, mi) do
        compilations[] += 1
        source_ir = mi.def.source::Expr
        deps = Core.MethodInstance[]
        result = interpret(cache, source_ir, deps)

        ci = create_ci(cache, mi; deps)
        results(cache, ci).ir = result
        return ci
    end
    return results(cache, ci).ir
end

## high-level API

function call(f, args...)
    tt = Tuple{map(Core.Typeof, args)...}
    world = Base.get_world_counter()
    mi = @something(method_instance(f, tt; world, method_table),
                    throw(MethodError(f, args)))

    cache = CacheView{ForeignResults}(:ForeignExample, world)
    compile!(cache, mi)
end


## demo

# Define a function with Expr IR (just returns 42)
function myop end
add_method(method_table, myop, (), Expr(:*, 21, 2))

# First call compiles
result = call(myop)  # argument value unused, IR has the value
@assert result == 42
@assert compilations[] == 1

# Second call uses cache
result = call(myop)
@assert result == 42
@assert compilations[] == 1

# Redefine with different IR - invalidates cache
add_method(method_table, myop, (), Expr(:^, 7, 2))
result = call(myop)
@assert result == 49
@assert compilations[] == 2

# Repeated call still cached
result = call(myop)
@assert result == 49
@assert compilations[] == 2

# Add method for different argument type - doesn't invalidate existing
add_method(method_table, myop, (Float64,), Expr(:*, 3.14, 1))
result = call(myop)  # Int version still cached
@assert result == 49
@assert compilations[] == 2

# Call the Float64 version - triggers new compilation
result = call(myop, 0.0)
@assert result == 3.14
@assert compilations[] == 3

# Both versions now cached
result = call(myop)
@assert result == 49
result = call(myop, 0.0)
@assert result == 3.14
@assert compilations[] == 3

println("Basic assertions passed!")


## Transitive dependency example
#
# grandchild_node <- child_node <- parent_node
#
# When grandchild is redefined, both child and parent should be invalidated.

println("\n--- Transitive Dependency Demo ---")
compilations[] = 0


# grandchild: base function, returns 1
function grandchild_node end
add_method(method_table, grandchild_node, (), Expr(:*, 1, 1))

# child: calls grandchild, multiplies by 2
function child_node end
add_method(method_table, child_node, (),
           Expr(:*, Expr(:call, grandchild_node), 2))

# parent: calls child, squares the result
function parent_node end
add_method(method_table, parent_node, (),
           Expr(:^, Expr(:call, child_node), 2))

# Compile all three
result = call(grandchild_node)
@assert result == 1  # 1 * 1
@assert compilations[] == 1
println("grandchild compiled: $result (compilations: $(compilations[]))")

result = call(child_node)
@assert result == 2  # grandchild() * 2 = 1 * 2
@assert compilations[] == 2  # child compiles, grandchild cached
println("child compiled: $result (compilations: $(compilations[]))")

result = call(parent_node)
@assert result == 4  # child()^2 = 2^2
@assert compilations[] == 3  # parent compiles, child+grandchild cached
println("parent compiled: $result (compilations: $(compilations[]))")

# Cache hits - no recompilation
call(grandchild_node)
call(child_node)
call(parent_node)
@assert compilations[] == 3
println("All cached (compilations still: $(compilations[]))")

# Now redefine grandchild to return 100
println("\nRedefining grandchild...")
add_method(method_table, grandchild_node, (), Expr(:*, 100, 1))

# grandchild should recompile
result = call(grandchild_node)
@assert result == 100
@assert compilations[] == 4
println("grandchild recompiled: $result (compilations: $(compilations[]))")

# child should recompile due to dependency on grandchild
result = call(child_node)
@assert result == 200  # 100 * 2
@assert compilations[] == 5
println("child recompiled: $result (compilations: $(compilations[]))")

# parent should recompile due to transitive dependency via child
result = call(parent_node)
@assert result == 40000  # 200^2
@assert compilations[] == 6
println("parent recompiled: $result (compilations: $(compilations[]))")

# All cached again
call(grandchild_node)
call(child_node)
call(parent_node)
@assert compilations[] == 6
println("All cached again (compilations still: $(compilations[]))")

println("\nAll transitive dependency assertions passed!")
