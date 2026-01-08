def interpret_objects(labels):
    if not labels:
        return None

    if "apple" in labels:
        return "That looks like an apple 🍎"

    if "bench" in labels:
        return "This is a bench 🪑"

    if "person" in labels:
        return "I see a person"

    return f"I think this is a {labels[0]}"

