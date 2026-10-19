"""Summary of the automatically resumed ML plan."""
def plan_summary(operations):
    count = len(operations)
    sources = len({tuple(op['dataset_key']) for op in operations})
    warning = ' · Большой пакет: проверьте объём перед запуском' if count >= 100 else ''
    return f'1 задача · {sources} наборов KNN · {count} операций ML{warning}'
