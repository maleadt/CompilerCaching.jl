# CompilerCaching.jl

A package for interfacing with Julia's compiler caching infrastructure for the purpose
of building custom compilers. It extends the existing `InternalCodeCache` type with
auxiliary functionality.


## Installation

```julia
using Pkg
Pkg.add(url="path/to/CompilerCaching")
```

## Basic usage

Julia's code caches are indexed with method instances, yielding a code instance that keeps
track of compilation results. Code instances are owned by a specific compiler, identified
by an owner token, and they contain a cache of results specific to that compiler.

The basic usage pattern of working with the compiler cache through CompilerCaching.jl:
1. Define a mutable results struct with a zero-arg constructor
2. Create a `CacheView{V}(owner, world)` where `V` is your results struct type
3. Use the cache's `Dict` interface to get or create a code instance for a method instance
4. Access compilation results via `results(cache, ci)`, populating them as needed

```julia
using CompilerCaching

# Define your results struct
mutable struct MyResults
    ir::Any
    code::Any
    executable::Any
    MyResults() = new(nothing, nothing, nothing)
end

# Compile a method instance
function compile(f, tt)
    world = Base.get_world_counter()
    mi = method_instance(f, tt; world)
    cache = CacheView{MyResults}(:MyCompiler, world)

    ci = get!(cache, mi) do
        create_ci(cache, mi)
    end
    res = results(cache, ci)

    if res.executable === nothing
        if res.ir === nothing
            res.ir = emit_ir(cache, mi)
        end
        if res.code === nothing
            res.code = emit_code(cache, mi, res.ir)
        end
        res.executable = emit_executable(cache, mi, res.code)
    end

    return res.executable
end
```

For compilers that use Julia's type inference, you'll also need an AbstractInterpreter
with `@setup_caching`:

```julia
# Set-up a custom interpreter, and link it to the cache
struct CustomInterpreter <: CC.AbstractInterpreter
    cache::CacheView
    ...
end
@setup_caching CustomInterpreter.cache

function compile(f, tt)
    # ...

    # Use typeinf! to populate the cache via inference
    ci = get(cache, mi, nothing)
    if ci === nothing
        interp = CustomInterpreter(cache)
        ir = CompilerCaching.typeinf!(cache, interp, mi)
        ci = ir[1][1]               # XXX: return the codeinstance
        results(cache, ci).ir = ir  # XXX: for use by codegen
    end

    # ...
end
```

The `@setup_caching` macro defines the necessary methods to connect the interpreter
to the cache:
- `CC.cache_owner(interp)` returning the cache's owner token
- `CC.finish!(interp, caller, ...)` that stacks a new `V()` instance in analysis results


## Cache sharding

It is possible to partition the cache by additional parameters by using a tuple or
named tuple as the owner key type:

```julia
function call(f, args...; opt_level=1)
    tt = map(Core.Typeof, args)
    world = Base.get_world_counter()
    mi = @something(method_instance(f, tt; world),
                    throw(MethodError(f, args)))

    cache = CacheView{MyResults}((:MyCompiler, opt_level), world)

    ci = get!(cache, mi) do
        create_ci(cache, mi)
    end
    res = results(cache, ci)
    # ... compile if needed
end
```

Different calls with the same owner key will hit the same cache partition.


## Overlay methods

It is often useful to redefine existing methods for use with the custom compiler.
This can be accomplished using overlay methods in a custom method table:

```julia
Base.Experimental.@MethodTable method_table
Base.Experimental.@overlay method_table function Base.sin(x::Int)
    # custom implementation
end

# Expose the method table to the interpreter
struct CustomInterpreter <: CC.AbstractInterpreter
    cache::CacheView
    ...
end
CC.method_table(interp::CustomInterpreter) = CC.OverlayMethodTable(interp.cache.world, method_table)

function call(f, args...)
    tt = map(Core.Typeof, args)
    world = Base.get_world_counter()
    mi = @something(method_instance(f, tt; world, method_table),
                    # if needed, look for global methods too
                    throw(MethodError(f, args)))

    cache = CacheView{MyResults}(:MyCompiler, world)
    ci = get!(cache, mi) do
        create_ci(cache, mi)
    end
    res = results(cache, ci)
    # ... compile if needed
end
```

If multiple overlay tables are needed, they can be stacked using `StackedMethodTable`:

```julia
MyMethodTableStack(world) = StackedMethodTable(world, overlay_table, base_table)

struct CustomInterpreter <: CC.AbstractInterpreter
    world::UInt
end
CC.method_table(interp::CustomInterpreter) = MyMethodTableStack(interp.world)
```


## Foreign IR

For compilers that define their own IR format that Julia doesn't know about, we cannot rely
on inference to populate the cache, so we need to bring our own code instances using
`create_ci`:

```julia
Base.Experimental.@MethodTable method_table

# Results struct for foreign IR
mutable struct ForeignResults
    ir::Any
    ForeignResults() = new(nothing)
end

# Only define our special functions in the overlay method table,
# providing our custom IR as the source.
function really_special end
add_method(method_table, really_special, (Int,), MyCustomIR([:a, :b]))

# Compile function using get! do-block pattern
function compile!(cache, mi)
    ci = get!(cache, mi) do
        ir = mi.def.source::MyCustomIR
        deps = Core.MethodInstance[]

        for callee in ir.callees
            callee_mi = method_instance(callee.f, callee.tt; world=cache.world, method_table)
            compile!(cache, callee_mi)  # recursive compilation
            push!(deps, callee_mi)
        end

        ci = create_ci(cache, mi; deps)
        results(cache, ci).ir = ir
        return ci
    end
    return results(cache, ci).ir
end

function call(f, args...)
    tt = Tuple{map(Core.Typeof, args)...}
    world = get_world_counter()
    mi = @something(method_instance(f, tt; world, method_table),
                    throw(MethodError(f, args)))

    cache = CacheView{ForeignResults}(:MyCompiler, world)
    compile!(cache, mi)
end
```
