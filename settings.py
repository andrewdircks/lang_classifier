DEV_DB_PATH = "snippets-dev/snippets-dev.db"

langs = [
    "Bash",
    "C",
    "C++",
    "CSV",
    "DOTFILE",
    "Go",
    "HTML",
    "JSON",
    "Java",
    "JavaScript",
    "Jupyter",
    "Markdown",
    "PowerShell",
    "Python",
    "Ruby",
    "Rust",
    "Shell",
    "TSV",
    "Text",
    "UNKNOWN",
    "YAML"
]

langs_ace = [
    "sh",
    "c_cpp",
    "c_cpp",
    "plain_text",
    "dot",
    "golang",
    "html",
    "json",
    "java",
    "javascript",
    "python", # jupyter is nothing
    "markdown",
    "powershell",
    "python",
    "ruby",
    "rust",
    "sh",
    "plain_text", # tab-seperated values
    "plain_text",
    "plain_text", # unknown
    "yaml"

]

def _dictify(seq):
    return {s: i for i, s in enumerate(seq)}

langs_map = _dictify(langs)
n_langs = len(langs)

PERCENT_TRAIN = 0.85