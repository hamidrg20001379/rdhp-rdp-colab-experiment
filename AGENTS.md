# Agent Notes

## Notebook JSON escaping

When editing `.ipynb` files directly, remember that the notebook is a JSON document, not just a Python source file. JSON does not allow Python-style escapes such as `\'`; a backslash before a single quote causes Colab to report `SyntaxError: Bad escaped character in JSON`.

Use unescaped single quotes inside JSON strings, and escape only characters required by JSON, such as double quotes and backslashes. After editing a notebook, validate it with a strict JSON parser before committing:

```sh
python -c "import json; json.load(open('notebook.ipynb', encoding='utf-8')); print('strict JSON OK')"
```
