import re
p="/anvil/scratch/x-jwang120/coastal-et/src/build_notebooks.py"
s=open(p).read()
def conv(text):
    text=re.sub(r"\bI've\b","we've",text); text=re.sub(r"\bI'll\b","we'll",text)
    text=re.sub(r"\bI'm\b","we're",text); text=re.sub(r"\bI'd\b","we'd",text)
    text=re.sub(r"\bMy\b","Our",text); text=re.sub(r"\bmy\b","our",text)
    text=re.sub(r"(^|[\n>]|\. |\.\n|- |\* |: |\*\*)I ", lambda m: m.group(1)+"We ", text)
    text=re.sub(r"\bI \b","we ",text)
    return text
def repl(m): return '("md", """' + conv(m.group(1)) + '""")'
s=re.sub(r'\("md", """(.*?)"""\)', repl, s, flags=re.S)
open(p,"w").write(s)
print("converted to we/our voice")
