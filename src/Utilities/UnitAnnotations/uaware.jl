using Cassette

export UnitCtx

Cassette.@Context UnitCtx

macro uaware(expr)
  # Evaluate both modified and unmodified input, but don't return any exprs
  Base.eval(__module__, expr)
  Base.eval(__module__, :(Cassette.@overdub(UnitCtx(), $expr)))
  nothing
end
