#==============================================================================#
# StackedMethodTable - composite method table view
#==============================================================================#

export StackedMethodTable

"""
    StackedMethodTable{MTV<:CC.MethodTableView} <: CC.MethodTableView

A composite method table view that stacks a custom method table with a parent view.
Lookups check the custom MT first, falling back to the parent if no fully-covering
match is found.

# Constructors
- `StackedMethodTable(world, mt::MethodTable)` - Stack `mt` over global MT
- `StackedMethodTable(world, mt::MethodTable, parent::MethodTableView)` - Stack `mt` over `parent`

# Usage
Return from `CC.method_table(interp::YourInterpreter)` in your AbstractInterpreter.
"""
struct StackedMethodTable{MTV<:CC.MethodTableView} <: CC.MethodTableView
    world::UInt
    mt::Core.MethodTable
    parent::MTV
end

# Convenience constructors
StackedMethodTable(world::UInt, mt::Core.MethodTable) =
    StackedMethodTable(world, mt, CC.InternalMethodTable(world))
StackedMethodTable(world::UInt, mt::Core.MethodTable, parent::Core.MethodTable) =
    StackedMethodTable(world, mt, StackedMethodTable(world, parent))

CC.isoverlayed(::StackedMethodTable) = true

# CC.findall - find all matching methods
function CC.findall(@nospecialize(sig::Type), table::StackedMethodTable; limit::Int=-1)
    result = CC._findall(sig, table.mt, table.world, limit)
    result === nothing && return nothing
    nr = CC.length(result)
    if nr >= 1 && CC.getindex(result, nr).fully_covers
        return result
    end

    parent_result = CC.findall(sig, table.parent; limit)::Union{Nothing, CC.MethodLookupResult}
    parent_result === nothing && return nothing

    return CC.MethodLookupResult(
        CC.vcat(result.matches, parent_result.matches),
        CC.WorldRange(
            CC.max(result.valid_worlds.min_world, parent_result.valid_worlds.min_world),
            CC.min(result.valid_worlds.max_world, parent_result.valid_worlds.max_world)),
        result.ambig | parent_result.ambig)
end

# CC.findsup - find most specific matching method
function CC.findsup(@nospecialize(sig::Type), table::StackedMethodTable)
    match, valid_worlds = CC._findsup(sig, table.mt, table.world)
    match !== nothing && return match, valid_worlds
    parent_match, parent_valid_worlds = CC.findsup(sig, table.parent)
    return (
        parent_match,
        CC.WorldRange(
            max(valid_worlds.min_world, parent_valid_worlds.min_world),
            min(valid_worlds.max_world, parent_valid_worlds.max_world)))
end
