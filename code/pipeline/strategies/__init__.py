"""Judgment strategies for the evidence-review pipeline.

Each strategy module exposes a function that takes a Claim plus its loaded
context (images, user history, evidence requirements) and returns a fully
assembled, schema-shaped output row dict via schema.build_output_row(...).

Strategies must stay fully general: no branching on case_id, user_id, or
any other case-specific identifier.
"""
