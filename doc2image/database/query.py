from .schema import Base, Session


def database_session_decorator(func):
    """
    Decorator to manage database sessions.

    Args:
        func (callable): The function to decorate.

    Returns:
        callable: The decorated function.
    """

    def wrapper(*args, **kwargs):
        with Session.begin() as session:
            result = func(session, *args, **kwargs)
            session.commit()
            return result

    return wrapper
