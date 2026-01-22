# CompilerCaching.jl

A package for interfacing with Julia's compiler caching infrastructure for the purpose
of building custom compilers. It extends the existing `InternalCodeCache` type with
auxiliary functionality.


## Installation

```julia
using Pkg
Pkg.add(url="path/to/CompilerCaching")
```


## Usage

The basic usage pattern is to create a `CacheView` and invoke `cached_compilation`
with a method instance and three callbacks:
- `emit_ir`: Analyze source code and emit high-level IR, populating the cache with code instances.
- `emit_code`: Generate serializable code that can be cached across sessions (e.g. LLVM IR).
- `emit_executable`: Generate a session-specific representation (e.g., a function pointer).

Or, in code:
```julia
using CompilerCaching

# Set-up a custom interpreter, and link it to the cache
struct CustomInterpreter <: CC.AbstractInterpreter
    cache::CacheView
    ...
end
CustomInterpreter(cache::CacheView) = ...
@setup_caching CustomInterpreter.cache

# Let Julia populate the cache and return codeinfos for emit_code
function emit_ir(cache, mi)
    interp = CustomInterpreter(cache)
    return CompilerCaching.populate!(cache, interp, mi)
end

# generate some code representation
function emit_code(cache, mi, ir) end

# compile code to function pointer
function emit_executable(cache, mi, code) end

function call(f, args...)
    tt = map(Core.Typeof, args)
    world = Base.get_world_counter()
    mi = @something(method_instance(f, tt; world),
                    throw(MethodError(f, args)))

    cache = CacheView(:MyCompiler, world)
    ptr = cached_compilation(cache, mi; emit_ir, emit_code, emit_executable)
    ccall(ptr, ...)
end

my_function(x::Int) = x + 100
call(my_function, 42)
```

The `@setup_caching` macro defines the necessary methods to connect the interpreter
to the cache, and can be customized further if needed.


### Cache sharding

It is possible to partition the cache by additional parameters by storing `keys` in the
cache view, e.g., encoding the optimization level or target device:

```julia
const CacheKey = @NamedTuple{opt_level::Int}

function call(f, args...; opt_level=1)
    tt = map(Core.Typeof, args)
    world = Base.get_world_counter()
    mi = @something(method_instance(f, tt; world),
                    throw(MethodError(f, args)))

    cache = CacheView{CacheKey}(:MyCompiler, world, (; opt_level))
    cached_compilation(cache, mi; emit_ir, emit_code, emit_executable)
end
```

Different calls with the same `(tag, keys)` will hit the same cache partition.


### Overlay methods

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

    cache = CacheView(:MyCompiler, world)
    cached_compilation(cache, mi; emit_ir, emit_code, emit_executable)
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


### Foreign IR

For compilers that define their own IR format that Julia doesn't know about, we cannot rely
on inference to populate the cache. Instead of calling `populate!`, we need to create our
own CodeInstances with `create_ci` and store them in the cache, providing dependencies to
ensure backedges are correctly set-up:

```julia
Base.Experimental.@MethodTable method_table

# Only define our special functions in the overlay method table,
# providing our custom IR as the source.
function really_special end
add_method(method_table, really_special, (Int,), MyCustomIR([:a, :b]))

# Since we're not relying on Julia's inference, we need to create and cache our own CIs.
# Recursive invocation of the underlying `get_ir` ensures dependencies are cached too.
function emit_ir(cache, mi)
    ir = mi.def.source::MyCustomIR
    deps = Core.MethodInstance[]
    for callee in ir.callees
        callee_mi = method_instance(callee.f, callee.tt; world=cache.world, method_table)
        callee_ci, callee_ir = get_ir(cache, callee_mi; emit_ir)
        # do something with the callee's IR if needed
        push!(deps, callee_mi)
    end
    cache[mi] = create_ci(cache, mi; deps)
    return ir
end

function emit_code(cache, mi, ir) end

function emit_executable(cache, mi, code) end

function call(f, args...)
    tt = Tuple{map(Core.Typeof, args)...}
    world = get_world_counter()
    mi = @something(method_instance(f, tt; world, method_table),
                    throw(MethodError(f, args)))

    cache = CacheView(:MyCompiler, world)
    cached_compilation(cache, mi; emit_ir, emit_code, emit_executable)
end
```
