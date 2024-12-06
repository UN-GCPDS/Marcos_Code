def multi_sort(*args:tuple,
                reverse : bool=True,
                ) -> tuple:
    """Sort multiple array-likes based on the first.

    Parameters
    ----------
    reverse : bool, optional
        Whether to sort in descending (True) or ascending order (False), by default True

    Returns
    -------
    tuple
        a tuple of ordered lists

    Examples
    -------
    >>> a = [2, 3, 1]
    >>> b = ['a', 'b', 'c']
    >>> print(multi_sort(a,b))
    ([3, 2, 1], ['b', 'a', 'c'])

    >>> arrays = [[2, 3, 1], ['a', 'b', 'c']]
    >>> print(multi_sort(*arrays))
    ([3, 2, 1], ['b', 'a', 'c'])
    """
    sorted_lists = (list(t) for t in zip(*sorted(zip(*args), reverse=reverse)))
    return tuple(sorted_lists)

