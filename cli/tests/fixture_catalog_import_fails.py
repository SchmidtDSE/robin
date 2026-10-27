"""A catalog adapter module that fails when it is imported, with a credential in the error."""

raise RuntimeError("token=abc123")
