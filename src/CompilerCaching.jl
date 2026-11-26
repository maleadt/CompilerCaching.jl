# CompilerCaching.jl - Reusable package for compiler result caching
#
# Leverages Julia's Method/MethodInstance/CodeInstance infrastructure to provide:
# - Lazy compilation with caching
# - Type-based specialization and dispatch
# - Automatic invalidation when methods are redefined
# - Transitive dependency tracking
#
# Requires Julia 1.11+

module CompilerCaching

"""
    @public foo, bar

Declare `foo, bar` as public API. Lowers to `public foo, bar` on 1.11+ (where `public`
is keyword syntax) and to a no-op on 1.10. Lets the rest of the module use a single
form regardless of Julia version, without `Meta.parse` workarounds.
"""
macro public(symbols_expr)
    syms = symbols_expr isa Symbol ? [symbols_expr] :
           symbols_expr.head === :tuple ? [a isa Symbol ? a : a.args[1] for a in symbols_expr.args] :
           [symbols_expr.args[1]]
    if VERSION >= v"1.11.0-DEV.469"
        esc(Expr(:public, syms...))
    else
        nothing
    end
end

@static if VERSION >= v"1.11"

include("objcache.jl")
@public ObjCache

using Base.Experimental: @MethodTable
const CC = Core.Compiler

include("utils.jl")


#==============================================================================#
# CacheView structure
#==============================================================================#

export CacheView, results, lookup, seeded

"""
    CachedResult{V}

Wrapper identifying the results struct of type `V` on a CI's `analysis_results` chain.
Attached on first access (see [`results`](@ref)) or at CI creation (see
[`create_ci`](@ref)).
"""
struct CachedResult{V}
    inner::V
end

"""
    CacheView{K, V}

A cache into a cache partition at a specific world age. Serves as the main entry point
for cached compilation.

# Owner choice and lookup cost

`owner` is stored on every cached `CodeInstance` as its `.owner` field. Cache
lookups (`get(cache, mi)`) ccall `jl_rettype_inferred`, which compares
`ci.owner` against the requested owner using `jl_egal` — i.e. structural
equality for immutable values, identity (`===`) for mutable ones.

The owner is passed to the ccall as `Any`, which forces the JIT to box it on
every call when its concrete type is not already a heap-allocated reference.
For hot paths this matters:

| owner type                            | per `get(cache, mi)` |
| ------------------------------------- | -------------------- |
| `Symbol`                              |   ~2.5 ns, 0 allocs  |
| `struct` (immutable, e.g. NamedTuple) |  ~10 ns,  small box  |
| `Tuple{Symbol, NamedTuple{…}}`        |  ~38 ns, ~112 B box  |

Prefer a `Symbol` if you don't need to shard. Otherwise, an immutable struct
or tuple is the right shape — it boxes on each lookup, but `CodeInstance`s
loaded from a package image still resolve because `jl_egal` matches by
content.

Avoid `mutable struct` owners: `jl_egal` falls back to identity for mutable
types, so a `CodeInstance` deserialized from a package image will not match
a fresh runtime instance even with identical fields.
"""
struct CacheView{K, V}
    owner::K
    world::UInt
    CacheView{K,V}(owner, world::UInt) where {K,V} = new{K,V}(convert(K, owner), world)
end

CacheView{V}(owner::K, world::UInt) where {K,V} = CacheView{K,V}(owner, world)

"""
    cache_owner(cache::CacheView)

Returns the owner token for use as CodeInstance.owner.
"""
cache_owner(cache::CacheView) = cache.owner

"""
    results_type(cache::CacheView{K,V}) -> Type{V}

The results type addressed by this cache view.
"""
results_type(::CacheView{K,V}) where {K,V} = V


## results attachment
#
# Results structs are attached to a CodeInstance's `analysis_results` chain *lazily*, on
# first access through `results` / `lookup`. This keeps inference entirely results-free:
# interpreters need no CompilerCaching-specific hooks (no `CC.finish!` override), and a
# single CI can carry results for multiple independent consumers (distinct `V` types
# coexist on the chain).

# `analysis_results` is declared const on Julia 1.11–1.13 (the C runtime mutates it in
# place after optimization), so plain `setfield!` is rejected and — more subtly — repeated
# `getfield`s may legally be CSE'd. We mutate through `jl_set_nth_field` (the same way
# `Serialization` writes const fields during deserialization) and re-read through an
# opaque ccall when re-checking under the lock.
const ANALYSIS_RESULTS_FIELD = something(findfirst(==(:analysis_results),
                                                   fieldnames(Core.CodeInstance)))

read_analysis_results(ci::Core.CodeInstance) =
    ccall(:jl_get_nth_field_checked, Any, (Any, Csize_t), ci, ANALYSIS_RESULTS_FIELD-1)

# Lock serializing chain mutations. Attachment is rare (once per (CI, V) pair), so a
# single global lock suffices. Plain (lock-free) reads are safe: chain nodes are
# immutable and only ever prepended, and `jl_set_nth_field` stores with release
# semantics.
#
# The C runtime also writes this field, without taking our lock: `jl_fill_codeinst`
# (and `jl_update_codeinst`) overwrite it wholesale when inference finishes. That's
# only safe because those writes happen while the CI is still private to the inference
# engine (jl_fill_codeinst asserts min_world == 1 / max_world == 0, i.e.
# pre-publication). Corollary: only attach results to CIs that have been published to
# the integrated cache — never to a CI still being inferred.
const attach_lock = ReentrantLock()

@noinline function attach_results!(::Type{V}, ci::Core.CodeInstance) where V
    Base.@lock attach_lock begin
        # re-read and re-check under the lock: another task may have attached while we
        # were acquiring it (the ccall also defeats CSE with the pre-lock traversal)
        chain = read_analysis_results(ci)
        head = chain isa CC.AnalysisResults ? chain : CC.NULL_ANALYSIS_RESULTS
        node = head
        while isdefined(node, :next)
            node.result isa CachedResult{V} && return node.result::CachedResult{V}
            node = node.next
        end

        cached = CachedResult{V}(V())
        ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), ci,
              ANALYSIS_RESULTS_FIELD-1, CC.AnalysisResults(cached, head))
        return cached
    end
end

function find_results(::Type{V}, ci::Core.CodeInstance) where V
    CC.traverse_analysis_results(ci) do @nospecialize result
        result isa CachedResult{V} ? result : nothing
    end
end

@inline function get_results(::Type{V}, ci::Core.CodeInstance) where V
    cached = find_results(V, ci)
    cached === nothing && (cached = attach_results!(V, ci))
    return cached::CachedResult{V}
end

"""
    results(::Type{V}, ci::CodeInstance)::V
    results(cache::CacheView{K,V}, ci::CodeInstance)::V

Retrieve the typed results struct of type `V` from a CodeInstance, creating and
attaching a fresh `V()` on first access. The same instance is returned for every
subsequent call with the same `V`, for the lifetime of the CodeInstance (including
across precompilation, when both the CI and its results are serialized into the
package image).

Mutations to a results struct attached to a CodeInstance that was loaded from a
*different* package image do not persist beyond the current (pre)compilation session;
only the image that serialized the CI owns its storage.
"""
results(::Type{V}, ci::Core.CodeInstance) where V = get_results(V, ci).inner

results(::CacheView{K,V}, ci::Core.CodeInstance) where {K,V} = results(V, ci)

@static if VERSION >= v"1.14-"
    function code_cache(cache::CacheView)
        world_range = CC.WorldRange(cache.world)
        return CC.InternalCodeCache(cache_owner(cache), world_range)
    end
else
    function code_cache(cache::CacheView)
        cc = CC.InternalCodeCache(cache_owner(cache))
        return CC.WorldView(cc, cache.world)
    end
end

# Expose InternalCodeCache functionality
Base.haskey(cache::CacheView, mi::Core.MethodInstance) = CC.haskey(code_cache(cache), mi)
Base.get(cache::CacheView, mi::Core.MethodInstance, default) = CC.get(code_cache(cache), mi, default)
function Base.get(cache::CacheView, mi::Core.MethodInstance)
    ci = get(cache, mi, nothing)
    ci === nothing && throw(KeyError(mi))
    return ci
end
Base.getindex(cache::CacheView, mi::Core.MethodInstance) = CC.getindex(code_cache(cache), mi)
Base.setindex!(cache::CacheView, ci::Core.CodeInstance, mi::Core.MethodInstance) = CC.setindex!(code_cache(cache), ci, mi)

"""
    lookup(cache::CacheView{K,V}, mi::MethodInstance) -> Union{Nothing, Tuple{CodeInstance, V}}

Single-pass cache lookup, combining `get(cache, mi)` and `results(cache, ci)`: returns
`(ci, res)` when a `CodeInstance` is cached for `mi` (attaching a fresh `V()` on first
access), or `nothing` when there is none.

Pass the returned pair through compilation phases to avoid repeated lookups.
"""
@inline function lookup(cache::CacheView{K,V}, mi::Core.MethodInstance) where {K,V}
    ci = get(cache, mi, nothing)
    ci === nothing && return nothing
    return (ci, results(V, ci))
end

# The owner of const-seeded inference results: the owner of the cache they were seeded
# in, plus the seeds. Not public, so that no ordinary owner can collide with it.
struct SeededOwner
    owner::Any
    argtypes::Tuple
end

"""
    seeded(cache::CacheView{K,V}, argtypes::Vector{Any}) -> CacheView

The cache view holding the results of const-seeded inference on `argtypes` (see
[`typeinf!(interp, mi, argtypes)`](@ref)) in `cache`'s partition and world. Every seeded
result is a `CodeInstance` of its own, so the usual operations apply: `get(view, mi,
nothing)` finds it, and `results(view, ci)` returns its results struct.

`argtypes` are compared by `===`, element by element, and copied, so the caller may
reuse the vector. Seeds must therefore be immutable to be found again: a `Const`
wrapping a mutable object only matches that very object, and a `PartialStruct`, which
holds vectors, only matches itself. Seeds that are constructed anew for every call
never match, so every call publishes another `CodeInstance`.

Lookups pass over seeded `CodeInstance`s without source (e.g. because a package image
dropped it): inferring the seeds again could give code with other dependencies than the
`CodeInstance` tracks, so [`typeinf!`](@ref typeinf!(::CC.AbstractInterpreter,
::Core.MethodInstance, ::Vector{Any})) publishes a new one instead. With
`CACHE_MODE_ROOT`, if the interpreter's codegen cache still holds the source, `typeinf!`
restores it and returns the same `CodeInstance`.
"""
seeded(cache::CacheView{K,V}, argtypes::Vector{Any}) where {K,V} =
    CacheView{SeededOwner,V}(SeededOwner(cache.owner, Tuple(argtypes)), cache.world)

function Base.get(cache::CacheView{SeededOwner}, mi::Core.MethodInstance, default)
    isdefined(mi, :cache) || return default
    ci = @atomic :acquire mi.cache
    while true
        if ci.owner === cache.owner &&
           (@atomic :acquire ci.min_world) <= cache.world <= (@atomic :acquire ci.max_world) &&
           seeded_source_available(ci)
            return ci
        end
        isdefined(ci, :next) || return default
        ci = @atomic :acquire ci.next
    end
end
Base.haskey(cache::CacheView{SeededOwner}, mi::Core.MethodInstance) =
    get(cache, mi, nothing) !== nothing
Base.getindex(cache::CacheView{SeededOwner}, mi::Core.MethodInstance) = get(cache, mi)

seeded_source_available(ci::Core.CodeInstance) =
    has_source(@atomic :monotonic ci.inferred) ||
    # `get_source` synthesizes the source of a const-return CodeInstance
    (@static VERSION >= v"1.12-" ? CC.use_const_api(ci) : false)

#==============================================================================#
# Cache access
#==============================================================================#

"""
    Base.get!(f::Function, cache::CacheView, mi::MethodInstance) -> CodeInstance

Get an existing CodeInstance or create one using `f()`.

Standard dict interface: returns existing CI if found, otherwise calls `f()`
which must return a CodeInstance, stores it, and returns it.

Concurrent misses may call `f()` more than once. The first CodeInstance inserted
is returned to every caller; the other candidates are discarded.

# Example (foreign mode)
```julia
ci = get!(cache, mi) do
    create_ci(cache, mi; deps)
end
```
"""
# Serialize publication only; construction remains parallel across cache misses.
const insert_lock = ReentrantLock()

function Base.get!(f::Function, cache::CacheView, mi::Core.MethodInstance)
    ci = get(cache, mi, nothing)
    ci !== nothing && return ci
    ci = f()::Core.CodeInstance
    # Another task may have inserted meanwhile; its CodeInstance wins, since results
    # may already be attached to it.
    Base.@lock insert_lock begin
        existing = get(cache, mi, nothing)
        existing !== nothing && return existing
        cache[mi] = ci
    end
    return ci
end



#==============================================================================#
# Foreign method registration
#==============================================================================#

export add_method
@public captured_globals

"""
    captured_globals(source) -> iterable of GlobalRef

Return the `GlobalRef`s a foreign IR `source` captures. Override this for
your custom IR type to enable automatic binding-invalidation tracking; the
default returns none.

[`create_ci`](@ref) consults this hook to wire the referenced bindings into
the runtime's invalidation mechanism without the caller having to thread
them through manually.

Report only bindings the IR actually reads: over-reporting causes spurious
invalidations. The result must be stable for the lifetime of `source` —
edges are registered on every CI created for a method using this source.
"""
captured_globals(@nospecialize(source)) = ()

"""
    add_method(mt, sig, source) -> Method
    add_method(mt, f, arg_types, source) -> Method

Register a method with custom source IR in the cache's method table.

# Arguments
- `mt::Core.MethodTable` - The method table to add the method to
- `sig::Type{<:Tuple}` - Full call signature. The first parameter must be the
  singleton type of a `Function`; use `UnionAll` to bind static parameters.
- `f::Function`, `arg_types::Tuple` - Function and argument types for a method
  without static parameters
- `source` - Custom IR to store (any type)

If `source` captures global bindings, override [`captured_globals`](@ref)
for its type. The bindings are then registered as edges of every
[`create_ci`](@ref) call for this method, so the cached code is
invalidated whenever any of them is replaced.

# Returns
The created `Method` object.
"""
function add_method(mt::Core.MethodTable, @nospecialize(sig::Type{<:Tuple}), source)
    params = Base.unwrap_unionall(sig).parameters
    isempty(params) && throw(ArgumentError("method signature must include a function type"))
    Base.has_free_typevars(sig) &&
        throw(ArgumentError("method signature has free type variables"))

    ft = params[1]
    if !(ft isa DataType && isdefined(ft, :instance) && ft.instance isa Function)
        throw(ArgumentError("method signature must begin with a singleton Function type"))
    end
    f = ft.instance
    mod = parentmodule(f)

    m = ccall(:jl_new_method_uninit, Any, (Any,), mod)

    m.name = nameof(f)
    m.module = mod
    m.file = Symbol("foreign")
    m.line = Int32(0)
    m.sig = sig
    m.nargs = Int32(length(params))
    m.isva = CC.isvarargtype(params[end])
    m.nospecialize = UInt32(0)
    m.external_mt = mt
    m.slot_syms = ""
    m.source = source

    # For non-CodeInfo sources, mark the source as scanned *before* publishing
    # the method, so any concurrent retrieve_code_info / jl_scan_method_source_now
    # skips its CodeInfo-only scan (which would otherwise crash trying to
    # uncompress foreign IR). For CodeInfo sources the bit is left untouched so
    # Julia's own scan still runs.
    @static if VERSION >= v"1.12-"
        if !isa(source, Core.CodeInfo)
            @atomic m.did_scan_source |= 0x1
        end
    end

    ccall(:jl_method_table_insert, Cvoid, (Any, Any, Any), mt, m, nothing)

    return m
end

add_method(mt::Core.MethodTable, f::Function, arg_types::Tuple, source) =
    add_method(mt, Tuple{typeof(f), arg_types...}, source)


#==============================================================================#
# Method lookup
#==============================================================================#

export method_instance, has_matching_method, match_method_instance

# JuliaLang/julia#62001 specializes closed type-valued callees and arguments on
# `Core.TypeEgal` dispatch keys, making `Type{T}` elements non-dispatchable.
# `Base.signature_type` handles the callee (it takes a value), but not `Type{T}`
# spellings in a user-provided `tt`, so normalize those ourselves.
@static if isdefined(Core, :TypeEgal)
    @inline function dispatch_key(@nospecialize(t))
        if Base.isType(t)
            u = Base.type_parameter(t)
            Base.has_free_typevars(u) || return Core.TypeEgal{u}
        end
        return t
    end

    function normalize_signature(@nospecialize(sig))
        u = Base.unwrap_unionall(sig)::DataType
        return Base.rewrap_unionall(Tuple{map(dispatch_key, u.parameters)...}, sig)
    end

    signature_type(@nospecialize(f), @nospecialize(tt)) =
        normalize_signature(Base.signature_type(f, tt))
else
    const normalize_signature = identity
    const signature_type = Base.signature_type
end

"""
    match_method_instance(sig; world, method_table) -> Union{MethodInstance, Nothing}
    match_method_instance(f, tt; world, method_table) -> Union{MethodInstance, Nothing}

Look up a MethodInstance using method matching instead of cached dispatch lookup.
Pass either the full call signature `sig`, including the function self as its first
parameter, or a known function value `f` and argument types `tt`.

Unlike `method_instance`, this function accepts non-dispatch tuples (abstract
argument types) without crashing. Use this for compile-time analysis where
argument types may not be fully concrete.

Returns `nothing` if no unique matching method is found.
"""
function match_method_instance(@nospecialize(sig::Type{<:Tuple});
                               world::UInt=Base.get_world_counter(),
                               method_table::Union{Core.MethodTable,Nothing}=nothing)
    sig = normalize_signature(sig)
    matches = Base._methods_by_ftype(sig, method_table, 1, world)
    matches === nothing && return nothing
    length(matches) != 1 && return nothing
    if VERSION >= v"1.12-"
        return Base.specialize_method(matches[1]::Core.MethodMatch)
    else
        return CC.specialize_method(matches[1]::Core.MethodMatch)
    end
end

function match_method_instance(@nospecialize(f), @nospecialize(tt);
                               world::UInt=Base.get_world_counter(),
                               method_table::Union{Core.MethodTable,Nothing}=nothing)
    return match_method_instance(signature_type(f, tt); world, method_table)
end

"""
    has_matching_method(sig; world, method_table) -> Bool
    has_matching_method(f, tt; world, method_table) -> Bool

Return whether a method fully covers the call signature. Pass either the full
signature `sig`, including the function self as its first parameter, or a known
function value `f` and argument types `tt`.

This is an existence query, not a uniqueness query: it may return `true` when
[`match_method_instance`](@ref) returns `nothing` because several methods
intersect an abstract signature.
"""
function has_matching_method(@nospecialize(sig::Type{<:Tuple});
                             world::UInt=Base.get_world_counter(),
                             method_table::Union{Core.MethodTable,Nothing}=nothing)
    sig = normalize_signature(sig)
    match, = CC._findsup(sig, method_table, world)
    return match !== nothing
end

function has_matching_method(@nospecialize(f), @nospecialize(tt);
                             world::UInt=Base.get_world_counter(),
                             method_table::Union{Core.MethodTable,Nothing}=nothing)
    return has_matching_method(signature_type(f, tt); world, method_table)
end


# jl_get_specialization1 doesn't support custom method tables (hardcodes jl_nothing).
# Reimplement its pipeline (match → normalize → specialize) with method table support.
function _specialization1(@nospecialize(sig), world::UInt,
                          method_table::Union{Core.MethodTable,Nothing})
    matches = Base._methods_by_ftype(sig, method_table, 1, world)
    matches === nothing && return nothing
    length(matches) != 1 && return nothing
    match = matches[1]::Core.MethodMatch
    m = match.method
    ti = match.spec_types
    env = match.sparams
    @static if VERSION >= v"1.12-"
        tt = ccall(:jl_normalize_to_compilable_sig, Any, (Any, Any, Any, Cint),
                   ti, env, m, Cint(1))
    else # 1.11: extra jl_methtable_t* first param
        mt = ccall(:jl_method_get_table, Any, (Any,), m)
        tt = ccall(:jl_normalize_to_compilable_sig, Any, (Any, Any, Any, Any, Cint),
                   mt, ti, env, m, Cint(1))
    end
    tt === nothing && return nothing
    if tt !== ti
        pair = ccall(:jl_type_intersection_with_env, Any, (Any, Any),
                     tt, m.sig)::Core.SimpleVector
        env = pair[2]::Core.SimpleVector
    end
    return ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
                 (Any, Any, Any), m, tt, env)
end

# Before JuliaLang/julia#60718, `jl_method_lookup_by_tt` did not correctly cache overlay
# methods, causing lookups to fail or return stale global entries, so don't use the cache.
# Use jl_get_specialization1 instead, which uses jl_matching_methods (not cached dispatch)
# and returns compileable signatures (with proper vararg widening).
# The overlay lookup issue is fixed in 1.13.0-beta2, 1.12.5, and 1.11.9. On
# 1.14+, `Base.method_instance` returns the dispatch-cache MI, and the compiler
# normalizes that to a compilable MI later. Do the same here, since this API is
# documented to return the compilation target.
@static if (v"1.13.0-beta2" <= VERSION < v"1.14-" ||
            v"1.12.5" <= VERSION < v"1.13-" ||
            v"1.11.9" <= VERSION < v"1.12-")
    @inline function method_instance(@nospecialize(f), @nospecialize(tt);
                                     world::UInt=Base.get_world_counter(),
                                     method_table::Union{Core.MethodTable,Nothing}=nothing)
        Base.method_instance(f, tt; world, method_table)
    end
elseif VERSION >= v"1.14.0-DEV.2337"
    @inline function method_instance(@nospecialize(f), @nospecialize(tt);
                                     world::UInt=Base.get_world_counter(),
                                     method_table::Union{Core.MethodTable,Nothing}=nothing)
        sig = signature_type(f, tt)
        @assert isdispatchtuple(sig)
        mi = Base.method_instance(sig; world, method_table)
        mi === nothing && return nothing
        return ccall(:jl_normalize_to_compilable_mi, Any, (Any,), mi)::Core.MethodInstance
    end
elseif VERSION >= v"1.13-"
    # 3-arg jl_get_specialization1, returns jl_nothing on failure
    @inline function method_instance(@nospecialize(f), @nospecialize(tt);
                                     world::UInt=Base.get_world_counter(),
                                     method_table::Union{Core.MethodTable,Nothing}=nothing)
        sig = signature_type(f, tt)
        @assert isdispatchtuple(sig)
        if method_table === nothing
            mi = ccall(:jl_get_specialization1, Any, (Any, Csize_t, Cint),
                       sig, world, Cint(0))
            return mi === nothing ? nothing : mi::Core.MethodInstance
        else
            return _specialization1(sig, world, method_table)
        end
    end
elseif VERSION >= v"1.12-"
    # 3-arg jl_get_specialization1, returns NULL on failure
    @inline function method_instance(@nospecialize(f), @nospecialize(tt);
                                     world::UInt=Base.get_world_counter(),
                                     method_table::Union{Core.MethodTable,Nothing}=nothing)
        sig = signature_type(f, tt)
        @assert isdispatchtuple(sig)
        if method_table === nothing
            ptr = ccall(:jl_get_specialization1, Ptr{Cvoid}, (Any, Csize_t, Cint),
                        sig, world, Cint(0))
            return ptr == C_NULL ? nothing : unsafe_pointer_to_objref(ptr)::Core.MethodInstance
        else
            return _specialization1(sig, world, method_table)
        end
    end
else # 1.11: 5-arg jl_get_specialization1 (extra min_valid/max_valid out-params), returns NULL
    @inline function method_instance(@nospecialize(f), @nospecialize(tt);
                                     world::UInt=Base.get_world_counter(),
                                     method_table::Union{Core.MethodTable,Nothing}=nothing)
        sig = signature_type(f, tt)
        @assert isdispatchtuple(sig)
        if method_table === nothing
            min_valid = Ref{Csize_t}(1)
            max_valid = Ref{Csize_t}(typemax(Csize_t))
            ptr = ccall(:jl_get_specialization1, Ptr{Cvoid},
                        (Any, Csize_t, Ref{Csize_t}, Ref{Csize_t}, Cint),
                        sig, world, min_valid, max_valid, Cint(0))
            return ptr == C_NULL ? nothing : unsafe_pointer_to_objref(ptr)::Core.MethodInstance
        else
            return _specialization1(sig, world, method_table)
        end
    end
end

"""
    method_instance(f, tt; world, method_table) -> Union{MethodInstance, Nothing}

Look up the compileable MethodInstance for function `f` with argument types `tt`.

Uses `jl_get_specialization1` (or `Base.method_instance` on Julia ≥ 1.14) to return
a compileable specialization with proper vararg widening.
Requires `tt` to be a dispatch tuple (fully concrete argument types).
Use [`match_method_instance`](@ref) for compile-time lookups where types
may not be fully resolved.

Returns `nothing` if no matching method is found.
"""
method_instance


#==============================================================================#
# Populating the cache
#==============================================================================#

export typeinf!, create_ci, get_source, get_codeinfos

# Julia's inference engine keys foreign-owner reservations by object address, while
# the integrated cache compares owners with `egal`. Equal immutable owners can
# therefore infer and publish the same MethodInstance concurrently. Serialize root
# inference for one logical cache key, without blocking unrelated compilations.
mutable struct InferenceReservation
    owner::Any
    mi::Core.MethodInstance
    lock::ReentrantLock
    users::Int
end

const inference_reservations = InferenceReservation[]
const inference_reservations_lock = ReentrantLock()

function acquire_inference_reservation(@nospecialize(owner), mi::Core.MethodInstance)
    Base.@lock inference_reservations_lock begin
        for reservation in inference_reservations
            if reservation.owner === owner && reservation.mi === mi
                reservation.users += 1
                return reservation
            end
        end
        reservation = InferenceReservation(owner, mi, ReentrantLock(), 1)
        push!(inference_reservations, reservation)
        return reservation
    end
end

function release_inference_reservation(reservation::InferenceReservation)
    Base.@lock inference_reservations_lock begin
        reservation.users -= 1
        if reservation.users == 0
            i = findfirst(item -> item === reservation, inference_reservations)::Int
            deleteat!(inference_reservations, i)
        end
    end
    return
end

function with_inference_reservation(f, @nospecialize(owner), mi::Core.MethodInstance)
    reservation = acquire_inference_reservation(owner, mi)
    locked = false
    try
        lock(reservation.lock)
        locked = true
        return f()
    finally
        locked && unlock(reservation.lock)
        release_inference_reservation(reservation)
    end
end

# Run `f()` with an empty local inference cache, then restore the caller's
# entries. The cache is either a `Vector{InferenceResult}` or, on newer Julia
# versions, an `InferenceCache` with a separate index.
function with_fresh_inference_cache(f, interp::CC.AbstractInterpreter)
    cache = CC.get_inference_cache(interp)
    @static if isdefined(Core.Compiler, :InferenceCache)
        if cache isa CC.InferenceCache
            results, index = copy(cache.results), copy(cache.index)
            empty!(cache.results)
            empty!(cache.index)
            try
                return f()
            finally
                copy!(cache.results, results)
                copy!(cache.index, index)
            end
        end
    end
    saved = copy(cache)
    empty!(cache)
    try
        return f()
    finally
        copy!(cache, saved)
    end
end

function infer_source!(interp::CC.AbstractInterpreter, ci::Core.CodeInstance)
    mi = ci_mi(ci)

    # Source stored in a CI is shared with everything that uses its cache, so it has to
    # come from that cache's own interpreter. Code inferred by `interp` can call CIs of
    # other owners, e.g. native ones for code it lets run natively.
    if ci.owner isa SeededOwner
        # inferring the seeds again could give code with other dependencies than `ci`
        # tracks (see `seeded`)
        return nothing
    elseif ci.owner !== CC.cache_owner(interp)
        ci.owner === nothing || return nothing
        interp = CC.NativeInterpreter(CC.get_inference_world(interp))
    end

    @static if isdefined(Core.Compiler, :find_cached_ci) &&
               isdefined(Core.Compiler, :codegen_cache) &&
               isdefined(Core.Compiler, :SOURCE_MODE_GET_SOURCE)
        # On Julia 1.14+, SOURCE_MODE_GET_SOURCE searches past source-less entries in
        # the integrated cache and gives interpreters with a codegen cache a fresh,
        # source-capable inference result when necessary. Persist that transient source
        # on the original CI: CompilerCaching relies on it surviving package images and
        # on keeping the CI identity to which analysis results may already be attached.
        codegen = CC.codegen_cache(interp)
        if codegen !== nothing
            source_ci = CC.typeinf_ext(interp, mi, CC.SOURCE_MODE_GET_SOURCE)
            if source_ci isa Core.CodeInstance && source_ci.rettype === ci.rettype
                src = get(codegen, source_ci, nothing)
                if src isa Core.CodeInfo
                    store_source!(ci, src)
                    return src
                end
            end
        end
    end

    # Older compilers and interpreters without a codegen cache need standalone
    # re-inference. Make it behave like a fresh `jl_typeinf` entry: tombstoned
    # (`LimitedAccuracy`) const-prop results left by a deeper root cascade would
    # otherwise suppress const-prop that succeeds in this shallower context
    # (JuliaGPU/CUDA.jl#3185).
    src = with_fresh_inference_cache(interp) do
        CC.typeinf_code(interp, mi, true)
    end
    src isa Core.CodeInfo && store_source!(ci, src)
    return src
end

# Without source, `inferred` holds `nothing` or, for a CI loaded from a package image on
# Julia 1.13+, the `UInt8` inlining cost that the serializer keeps after discarding it.
has_source(@nospecialize(inferred)) = inferred isa Core.CodeInfo || inferred isa String

# Store `src` on `ci`, unless another task already stored source there.
function store_source!(ci::Core.CodeInstance, src::Core.CodeInfo)
    old = @atomic :monotonic ci.inferred
    while !has_source(old)
        old, success = @atomicreplace ci.inferred old => src
        success && break
    end
    return
end

ci_mi(ci::Core.CodeInstance) =
    @static VERSION >= v"1.12-" ? CC.get_ci_mi(ci) : ci.def::Core.MethodInstance

# The optimized source of a finished inference result, if it has one.
function inferred_source(result::CC.InferenceResult)
    src = result.src
    src isa CC.OptimizationState && (src = CC.ir_to_codeinf!(src))
    return src isa Core.CodeInfo ? src : nothing
end

"""
    typeinf!(interp, mi) -> Union{CodeInstance, Nothing}

Run type inference on `mi` using `interp`, storing the resulting `CodeInstance`
in Julia's integrated cache (partitioned by `CC.cache_owner(interp)`). Eagerly
compiles all callees and stores their source so [`get_codeinfos`](@ref) works.

Returns the root `CodeInstance` (or `nothing` if inference failed). Subsequent
calls for the same `mi` and world are no-ops — the existing CI is returned.

The eager callee walk only follows `:invoke` edges that refer to a `CodeInstance`.
Optimized source (in particular source reused from the cache) can also contain
`:invoke` statements targeting a bare `MethodInstance`; those callees are not
compiled here, but are resolved lazily by [`get_codeinfos(interp, ci)`](@ref).
"""
function typeinf!(interp::CC.AbstractInterpreter, mi::Core.MethodInstance)
    @static if VERSION >= v"1.12.0-DEV.1434"
        # `mi` is inferred exactly as requested; callers wanting a compileable
        # specialization should normalize before using it as a cache key.
        ci = with_inference_reservation(CC.cache_owner(interp), mi) do
            CC.typeinf_ext(interp, mi, CC.SOURCE_MODE_NOT_REQUIRED)
        end
        ci === nothing && return nothing

        # If another task inferred `mi` concurrently and published first, Julia's
        # `cache_result!` skips our insert and `typeinf_ext` hands back the orphan.
        # Adopt the published CodeInstance, so results attach to the one every cache
        # lookup returns.
        cache = CacheView{Nothing}(CC.cache_owner(interp), CC.get_inference_world(interp))
        ci = @something get(cache, mi, nothing) ci

        compile_callees!(interp, ci)
        return ci
    elseif VERSION >= v"1.12.0-DEV.15"
        cache = CacheView{Nothing}(CC.cache_owner(interp), CC.get_inference_world(interp))
        inferred_ci = with_inference_reservation(CC.cache_owner(interp), mi) do
            CC.typeinf_ext_toplevel(interp, mi, CC.SOURCE_MODE_FORCE_SOURCE)
        end
        @assert inferred_ci !== nothing "Inference of $mi failed"

        # inference should have populated the cache
        ci = get(cache, mi, nothing)
        ci === nothing && return nothing

        # if ci is rettype_const, the inference result won't have been cached
        # (because it is normally not supposed to be used ever again).
        # to avoid the need to re-infer, set that field here.
        if ci.inferred === nothing
            cache[mi] = inferred_ci
        end
        return ci
    else
        # Julia 1.11: typeinf_ext_toplevel returns CodeInfo, not CI
        cache = CacheView{Nothing}(CC.cache_owner(interp), CC.get_inference_world(interp))
        src = with_inference_reservation(CC.cache_owner(interp), mi) do
            CC.typeinf_ext_toplevel(interp, mi)
        end
        @assert src !== nothing "Inference of $mi failed"

        # inference should have populated the cache
        ci = get(cache, mi, nothing)
        ci === nothing && return nothing

        # if ci is rettype_const, the inference result won't have been cached.
        # to avoid the need to re-infer, set that field here.
        @atomicreplace ci.inferred nothing => src
        return ci
    end
end

# Make sure `ci` and every callee reachable through its `:invoke` edges has source.
function compile_callees!(interp::CC.AbstractInterpreter, ci::Core.CodeInstance)
    @static if VERSION >= v"1.12.0-DEV.1434"
        has_compilequeue = VERSION >= v"1.13.0-DEV.499" || v"1.12-beta3" <= VERSION < v"1.13-"
        if has_compilequeue
            workqueue = CC.CompilationQueue(; interp)
            push!(workqueue, ci)
        else
            workqueue = Core.CodeInstance[ci]
            inspected = IdSet{Core.CodeInstance}()
        end

        while !isempty(workqueue)
            callee = pop!(workqueue)
            if has_compilequeue
                CC.isinspected(workqueue, callee) && continue
                CC.markinspected!(workqueue, callee)
            else
                callee in inspected && continue
                push!(inspected, callee)
            end

            # now make sure everything has source code, if desired
            callee_mi = CC.get_ci_mi(callee)
            if CC.use_const_api(callee)
                # const-return: get_source will synthesize CodeInfo, no need to store
                continue
            end

            # Reuse source already stored on the CI (by inference, or by a previous
            # walk) instead of unconditionally calling `typeinf_code`, which re-runs
            # inference and optimization without consulting the cache. This makes
            # repeated walks over an already-populated graph (e.g. `cached_results`
            # followed by the back-end's compile) traversal-only.
            src = get_source(callee)
            if src === nothing
                src = infer_source!(interp, callee)
            end
            if src isa Core.CodeInfo
                if has_compilequeue
                    sptypes = CC.sptypes_from_meth_instance(callee_mi)
                    CC.collectinvokes!(workqueue, src, sptypes)
                else
                    CC.collectinvokes!(workqueue, src)
                end
            end
        end
    else
        get_source(ci) === nothing && infer_source!(interp, ci)
    end
    return
end

"""
    typeinf!(interp, mi, argtypes::Vector{Any}) -> Union{CodeInstance, Nothing}

Run const-seeded type inference on `mi`. `argtypes` are extended-lattice argument types
as at a call site, starting with the callable and unpacked for a varargs method, e.g.
`Any[Core.Const(f), Int, Core.Const(2)]`.

The result is a `CodeInstance` of its own, with the edges and world range of this
inference, so redefining anything its source depends on invalidates it. It is published
in [`seeded(cache, argtypes)`](@ref), for the cache of `interp` (its
`CC.cache_owner` and `CC.get_inference_world`). Callees are cached under that owner, as
with [`typeinf!(interp, mi)`](@ref).

Returns the published `CodeInstance`, which has source, or `nothing` if inference failed.
Calling it again returns the same `CodeInstance`, unless that lost its source (see
[`seeded`](@ref)). Concurrent calls with equal seeds infer once.

On Julia 1.14 and later builds without root inference under a separate owner
(`CACHE_MODE_ROOT`, added in JuliaLang/julia#65237), the result is valid only in the
world it was inferred in, so every world age increment causes it to be inferred again.
"""
function typeinf!(interp::CC.AbstractInterpreter, mi::Core.MethodInstance,
                  argtypes::Vector{Any})
    cache = seeded(CacheView{Nothing}(CC.cache_owner(interp),
                                      CC.get_inference_world(interp)), argtypes)
    ci = get(cache, mi, nothing)
    ci === nothing || return ci
    ci = with_inference_reservation(cache.owner, mi) do
        published = get(cache, mi, nothing)
        published === nothing ? infer_seeded!(interp, mi, cache) : published
    end
    ci === nothing && return nothing
    compile_callees!(interp, ci)
    return ci
end

# A seeded root is inferred in a frame Julia does not publish, so that it can be
# published under the seeded owner. Julia 1.14 removed the volatile mode; a local frame
# leaves its CodeInstance unpublished in the same way.
const SEEDED_CACHE_MODE = isdefined(CC, :CACHE_MODE_VOLATILE) ? :volatile : :local

function seeded_inference_result(interp::CC.AbstractInterpreter, mi::Core.MethodInstance,
                                 argtypes::Vector{Any})
    𝕃 = CC.typeinf_lattice(interp)
    @static if VERSION >= v"1.12-"
        # as constant propagation does (`const_prop_call`)
        cache_argtypes = CC.matching_cache_argtypes(𝕃, mi)
        argtypes = CC.matching_cache_argtypes(𝕃, mi, CC.SimpleArgtypes(argtypes), cache_argtypes)
        overridden = falses(length(argtypes))
        for i in eachindex(argtypes)
            overridden[i] = argtypes[i] !== CC.argtype_by_index(cache_argtypes, i)
        end
        return CC.InferenceResult(mi, argtypes, overridden)
    else
        return CC.InferenceResult(mi, CC.SimpleArgtypes(argtypes), 𝕃)
    end
end

# Infer and publish a seeded root. The caller holds the reservation for `cache.owner`.
function infer_seeded!(interp::CC.AbstractInterpreter, mi::Core.MethodInstance,
                       cache::CacheView)
    owner = cache.owner
    result = seeded_inference_result(interp, mi, collect(Any, owner.argtypes))
    @static if isdefined(Core.Compiler, :CACHE_MODE_ROOT)
        # Julia infers, publishes and extends the root under its own owner. Requiring
        # source, it only returns a published CI if that has source, maybe only in the
        # codegen cache of `interp`.
        ci = CC.typeinf_ext(interp, result, owner, CC.SOURCE_MODE_GET_SOURCE)
        ci === nothing && return nothing
        if get_source(ci) === nothing
            codegen = CC.codegen_cache(interp)
            src = @something(inferred_source(result),
                             codegen === nothing ? nothing : get(codegen, ci, nothing),
                             Some(nothing))
            src isa Core.CodeInfo && store_source!(ci, src)
        end
        return ci
    elseif isdefined(Core.Compiler, :engine_reserve)
        # Julia fills in the CodeInstance of a frame that has one, stores its backedges,
        # and (before 1.14) extends it to later worlds.
        ci = CC.engine_reserve(mi, owner)
        result.ci = ci
        try
            frame = CC.InferenceState(result, SEEDED_CACHE_MODE, interp)
            frame === nothing && return nothing
            CC.typeinf(interp, frame)
        finally
            # finished inference has released the reservation already
            CC.engine_reject(interp, ci)
        end
        result.tombstone && return nothing
    else
        frame = CC.InferenceState(result, :volatile, interp)
        frame === nothing && return nothing
        CC.typeinf(interp, frame)
        # a limited result has no source
        result.src === nothing && return nothing
        ci = seeded_code_instance(interp, result, owner)
    end
    # The interpreter may have discarded the source, which the seeds cannot be
    # recovered from cheaply.
    if get_source(ci) === nothing
        src = inferred_source(result)
        src === nothing || store_source!(ci, src)
    end
    cache[mi] = ci
    @static if !isdefined(Core.Compiler, :engine_reserve)
        CC.track_newly_inferred.x && ccall(:jl_push_newly_inferred, Cvoid, (Any,), ci)
    end
    return ci
end

@static if !isdefined(Core.Compiler, :engine_reserve)
# Julia 1.11 constructs a CodeInstance with the interpreter's owner, and publishes it in
# `cache_result!`; neither takes another owner. Rebuild Julia's CodeInstance under the
# seeded owner, extending it to later worlds as `cache_result!` does: `finish!` has
# stored the backedges if the result is valid in the current world. Like there, a method
# definition between that check and the insertion into the cache goes unnoticed.
function seeded_code_instance(interp::CC.AbstractInterpreter, result::CC.InferenceResult,
                              owner::SeededOwner)
    valid_worlds = result.valid_worlds
    if valid_worlds.max_world == Base.get_world_counter()
        valid_worlds = CC.WorldRange(valid_worlds.min_world, typemax(UInt))
    end
    ci = CC.CodeInstance(interp, result, valid_worlds)
    has_const = isdefined(ci, :rettype_const)
    const_flags = Int32((has_const ? 0x2 : 0x0) | (CC.use_const_api(ci) ? 0x1 : 0x0))
    return Core.CodeInstance(ci.def, owner, ci.rettype, ci.exctype,
                             has_const ? ci.rettype_const : nothing, ci.inferred,
                             const_flags, ci.min_world, ci.max_world, ci.ipo_purity_bits,
                             ci.purity_bits, ci.analysis_results, ci.relocatability)
end
end

const CompilationDependency = Union{Core.MethodInstance,Core.CodeInstance}
@public CompilationDependency

"""
    create_ci(cache::CacheView{K,V}, mi; deps=[], edges=nothing) -> CodeInstance

Create a CodeInstance for `mi` with `cache.owner`, fresh `V()` results, and
backedges for automatic invalidation. Used for foreign IR where inference
does not run.

Pass dependencies either as `deps`, a vector of `CompilationDependency`
(`MethodInstance` or `CodeInstance`), or as `edges::Core.SimpleVector` in Julia's
inference edge encoding (`CodeInfo.edges` / `CodeInstance.edges`). The latter
also supports method-table dependencies, `invoke` signatures, and `Core.Binding`
entries. The caller must supply a valid edge list for the running Julia version.
Passing nonempty `deps` together with `edges` throws an `ArgumentError`.

On Julia 1.12 and newer, a `CodeInstance` dependency tracks that exact compilation;
a `MethodInstance` dependency tracks any compilation of that method. Binding
edges from [`captured_globals(mi.def.source)`](@ref captured_globals) are appended
to either form. All binding edges are registered directly against the new
CodeInstance, without scanning the source as Julia IR.

On Julia 1.11, both forms only register MethodInstance-to-MethodInstance
backedges: CodeInstances reduce to their MethodInstance, and signatures,
method tables, and bindings are ignored. In particular, adding a method that
matches a method-table edge does not invalidate the CodeInstance on 1.11.
"""
function create_ci(cache::CacheView{K,V}, mi::Core.MethodInstance;
                   deps::AbstractVector=CompilationDependency[],
                   edges::Union{Nothing,Core.SimpleVector}=nothing) where {K,V}
    owner = cache.owner
    world = cache.world

    if edges !== nothing && !isempty(deps)
        throw(ArgumentError("create_ci: pass either `deps` or a pre-encoded `edges` list, not both"))
    end
    # Julia's edge iterator interprets some non-dependency values as metadata.
    # Reject them in `deps`, which only accepts method or code instances.
    if edges === nothing
        for dep in deps
            dep isa CompilationDependency ||
                throw(ArgumentError("create_ci: `deps` entries must be a MethodInstance or CodeInstance, got $(typeof(dep))"))
        end
    end
    forward_edges = edges === nothing ? Core.svec(deps...) : edges

    @static if VERSION >= v"1.12-"
        binding_edges = Core.Binding[]
        if isa(mi.def, Core.Method) && isdefined(mi.def, :source)
            for e in captured_globals(mi.def.source)
                push!(binding_edges,
                      e isa Core.Binding ? e : convert(Core.Binding, e::GlobalRef))
            end
        end
        ci_edges = isempty(binding_edges) ? forward_edges :
            Core.svec(forward_edges..., binding_edges...)
    else
        # Julia 1.11 has no per-CI edges field
        ci_edges = forward_edges
    end

    # Create typed results instance via CachedResult{V}
    ar = CC.AnalysisResults(CachedResult{V}(V()), CC.NULL_ANALYSIS_RESULTS)

    @static if VERSION >= v"1.12-"
        ci = Core.CodeInstance(mi, owner, Any, Any, nothing, nothing,
            Int32(0), world, typemax(UInt), UInt32(0), ar, nothing, ci_edges)
    else
        ci = Core.CodeInstance(mi, owner, Any, Any, nothing, nothing,
            Int32(0), world, typemax(UInt), UInt32(0), UInt32(0), ar, UInt8(0))
    end

    # Register backedges for automatic invalidation
    @static if VERSION >= v"1.12-"
        # Same-module bindings must bypass Julia's source-scanning path, which
        # cannot recover supplied edges from foreign IR (or unrelated CodeInfo).
        if any(edge -> edge isa Core.Binding, forward_edges)
            method_edges = Core.svec(filter(edge -> !(edge isa Core.Binding),
                                            collect(forward_edges))...)
            CC.store_backedges(ci, method_edges)
        else
            CC.store_backedges(ci, forward_edges)
        end
        for edge in ci_edges
            edge isa Core.Binding || continue
            ccall(:jl_add_binding_backedge, Cvoid, (Any, Any), edge, ci)
        end
    else
        store_backedges(mi, forward_edges)
    end

    return ci
end

@static if VERSION < v"1.12-"
# Julia 1.11 can only track the method instances in the supplied edge list.
function store_backedges(mi::Core.MethodInstance, edges::Core.SimpleVector)
    isa(mi.def, Method) || return  # don't add backedges to toplevel

    for edge in edges
        if edge isa Core.CodeInstance
            edge = edge.def::Core.MethodInstance
        end
        edge isa Core.MethodInstance || continue
        ccall(:jl_method_instance_add_backedge, Cvoid,
              (Any, Any, Any), edge, nothing, mi)
    end
    nothing
end
end

"""
    get_source(ci::CodeInstance) -> Union{CodeInfo, Nothing}

Retrieve CodeInfo from a CodeInstance's inferred field.
Handles decompression if stored as String, and generates synthetic
CodeInfo for const-return functions.

Returns `nothing` if CodeInfo cannot be retrieved (e.g., for runtime
functions inferred by NativeInterpreter that don't store source).
For the root CI from `typeinf!`, this should always return valid CodeInfo.
"""
function get_source(ci::Core.CodeInstance)
    mi = @static if VERSION >= v"1.12-"
        CC.get_ci_mi(ci)
    else
        ci.def::Core.MethodInstance
    end

    src = @atomic :monotonic ci.inferred
    if !has_source(src)
        # For const-return functions, generate synthetic CodeInfo
        if CC.use_const_api(ci)
            @static if VERSION >= v"1.13.0-DEV.1121"
                src = CC.codeinfo_for_const(CC.NativeInterpreter(), mi,
                    CC.WorldRange(ci.min_world, ci.max_world),
                    ci.edges, ci.rettype_const)
            elseif VERSION >= v"1.12-"
                src = CC.codeinfo_for_const(CC.NativeInterpreter(), mi, ci.rettype_const)
                # Work around 1.12/1.13 not setting nargs/isva in `codeinfo_for_const`
                @static if v"1.12-" <= VERSION < v"1.14.0-DEV.60"
                    if src.nargs == 0 && mi.def isa Method
                        src.nargs = mi.def.nargs
                        src.isva = mi.def.isva
                    end
                end
            end
        end
    elseif src isa String
        # Decompress if stored as compressed String
        src = ccall(:jl_uncompress_ir, Ref{Core.CodeInfo},
                    (Any, Any, Any), mi.def::Method, ci, src)
    end
    return src isa Core.CodeInfo ? src : nothing
end

"""
    get_codeinfos(interp::AbstractInterpreter, ci::CodeInstance) ->
        Vector{Pair{CodeInstance, CodeInfo}}

Collect the `CodeInstance`/`CodeInfo` pairs needed to generate code for `ci`, walking
`:invoke` edges transitively from the root. On Julia 1.12+ the result is closed under
direct-call edges, making it suitable for closed-world code generation
(`jl_emit_native`, which cannot look up missing callees during codegen).

`interp` is used to repair gaps that cache history can leave behind:

- `:invoke`/`:invoke_modify` statements whose target is still a bare `MethodInstance`
  (inlining's `compileable_specialization` emits those when the compileable
  specialization was not cached at optimization time; codegen lowers them to runtime
  dispatch) are resolved to a `CodeInstance` — running inference through `interp` if
  the cache has none — and the statement is rewritten to target it, in a copy of the
  containing source; cached `CodeInfo` is never mutated. Only targets callable through
  a native ABI (concrete signature, fully-resolved sparams) are resolved; anything
  else (e.g. `@nospecialize`-widened compileable signatures) deliberately keeps its
  runtime-dispatch fallback semantics. Interpreters that emit such statements on
  purpose can opt out with [`resolve_dynamic_invokes`](@ref).
- Referenced `CodeInstance`s that lack stored source (e.g. cached by an earlier
  session) are re-inferred.

Callees owned by another cache (e.g. native code that `interp` lets run natively) are
collected too, unless `interp` opts out with [`collect_foreign_callees`](@ref).

`interp` must match the cache owner and world that produced `ci` (typically the
interpreter previously passed to [`typeinf!`](@ref)).

On Julia 1.11, code generation resolves callees through a lookup callback instead, so
only the root entry is returned.
"""
function get_codeinfos end

@public resolve_dynamic_invokes

"""
    resolve_dynamic_invokes(interp::AbstractInterpreter) -> Bool

Whether [`get_codeinfos`](@ref) turns `:invoke` statements that target a bare
`MethodInstance` into direct calls to a `CodeInstance` inferred with `interp`.
Defaults to `true`; overload it for an interpreter type to change that.

Codegen lowers an `:invoke` of a `MethodInstance` to a call into the runtime
(`jl_invoke`), which compiles and runs the callee with Julia's native compiler,
not with `interp`. Resolving such statements keeps the callee in the compiled
code, which is required when there is no runtime to call into (e.g. on GPUs).
An interpreter that emits these statements on purpose, so that the callee
runs natively, should return `false`: resolving them would compile the callee
with `interp` instead. The statements are then left to the runtime, and the
result of `get_codeinfos` remains closed under its direct-call edges.
"""
resolve_dynamic_invokes(::CC.AbstractInterpreter) = true

@public collect_foreign_callees

"""
    collect_foreign_callees(interp::AbstractInterpreter) -> Bool

Whether [`get_codeinfos`](@ref) includes callees whose `CodeInstance` is owned by another
cache than that of `interp`, such as native code that `interp` lets run natively. Defaults
to `true`; overload it for an interpreter type to change that.

Including them makes the result closed under its direct-call edges, which is required
when there is no runtime to call into (e.g. on GPUs), but it compiles the entire native
call tree of such a callee again. An interpreter whose code runs on the host can return
`false`: calls to these callees are then left to codegen, which calls the existing native
code through the runtime (`jl_invoke`).
"""
collect_foreign_callees(::CC.AbstractInterpreter) = true

function get_codeinfos(interp::CC.AbstractInterpreter, root::Core.CodeInstance)
    codeinfos = Pair{Core.CodeInstance, Core.CodeInfo}[]
    @static if VERSION >= v"1.12-"
        resolve = resolve_dynamic_invokes(interp)
        foreign = collect_foreign_callees(interp)
        visited = IdSet{Core.CodeInstance}()
        workqueue = Core.CodeInstance[root]
        while !isempty(workqueue)
            callee_ci = pop!(workqueue)
            callee_ci in visited && continue
            push!(visited, callee_ci)
            foreign || callee_ci === root ||
                callee_ci.owner === CC.cache_owner(interp) || continue

            src = get_source(callee_ci)
            if src === nothing
                # a referenced CI may lack stored source (e.g. it was cached by an
                # earlier session whose sources were dropped); re-establish it
                if callee_ci.owner === CC.cache_owner(interp)
                    typeinf!(interp, CC.get_ci_mi(callee_ci))
                else
                    infer_source!(interp, callee_ci)
                end
                src = get_source(callee_ci)
                # if inference cannot provide source either, leave the call site to
                # codegen's runtime-dispatch fallback
                src === nothing && continue
            end
            resolve && (src = resolve_invoke_targets(interp, src))
            push!(codeinfos, callee_ci => src)

            for stmt in src.code
                if stmt isa Expr && stmt.head === :(=)
                    stmt = stmt.args[2]
                end
                if stmt isa Expr && (stmt.head === :invoke || stmt.head === :invoke_modify)
                    callee = stmt.args[1]
                    if callee isa Core.CodeInstance
                        push!(workqueue, callee)
                    end
                end
            end
        end
    else
        src = @something get_source(root) infer_source!(interp, root) Some(nothing)
        src !== nothing && push!(codeinfos, root => src)
    end
    return codeinfos
end

@static if VERSION >= v"1.12-"
# Specializations with incomplete sparams cannot be called through a native specsig
# ABI: codegen's `needsparams` path emits `jl_invoke` even for CodeInstance operands,
# so rewriting such targets is useless. Delegate the structural checks to Compiler
# when available, but retain the explicit TypeVar check: codegen rejects free type
# variables while Compiler.has_valid_abi_sparams currently does not.
function has_valid_abi_sparams(mi::Core.MethodInstance)
    @static if isdefined(Core.Compiler, :has_valid_abi_sparams)
        CC.has_valid_abi_sparams(mi) || return false
        for sp in mi.sparam_vals
            sp isa TypeVar && return false
        end
        return true
    end

    for sp in mi.sparam_vals
        if sp isa TypeVar || sp isa Core.SimpleVector || CC.isvarargtype(sp)
            return false
        end
    end
    return true
end

# Rewrite `:invoke`/`:invoke_modify` statements whose target is still a bare
# `MethodInstance` to target a `CodeInstance` instead, inferring one when the cache
# has none. Codegen can only emit a direct call for a `CodeInstance` operand; a
# `MethodInstance` operand unconditionally lowers to runtime dispatch. Returns `src`
# unchanged when there is nothing to rewrite, or a rewritten copy (cached source is
# never mutated).
function resolve_invoke_targets(interp::CC.AbstractInterpreter, src::Core.CodeInfo)
    resolved = src
    for pc in eachindex(resolved.code)
        stmt = resolved.code[pc]
        rhs = stmt isa Expr && stmt.head === :(=) ? stmt.args[2] : stmt
        # separate checks, so that inference narrows `rhs` to an `Expr`
        rhs isa Expr || continue
        rhs.head === :invoke || rhs.head === :invoke_modify || continue

        callee_mi = rhs.args[1]
        callee_mi isa Core.MethodInstance || continue
        # Only specializations with a fully-concrete signature can be invoked directly
        # through a native ABI; anything else (e.g. `@nospecialize`-widened compileable
        # signatures) keeps its runtime-dispatch fallback semantics.
        callee_mi.def isa Method && isdispatchtuple(callee_mi.specTypes) &&
            has_valid_abi_sparams(callee_mi) || continue

        callee_ci = typeinf!(interp, callee_mi)
        callee_ci === nothing && continue
        # only rewrite when the callee's source is available, so the returned
        # collection stays closed under direct-call edges
        get_source(callee_ci) === nothing && continue

        if resolved === src
            resolved = copy(src)
            stmt = resolved.code[pc]
            rhs = (stmt isa Expr && stmt.head === :(=) ? stmt.args[2] : stmt)::Expr
        end
        new_rhs = Expr(rhs.head, callee_ci, rhs.args[2:end]...)
        resolved.code[pc] = stmt === rhs ? new_rhs : Expr(:(=), (stmt::Expr).args[1], new_rhs)
    end
    return resolved
end
end

end # @static if VERSION >= v"1.11"

end # module CompilerCaching
