with open('pdf_parser.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到被搞坏的行并修复
new_lines = []
for line in lines:
    if 'self.metadata = {' in line and '_extract_metadata()' in line:
        new_lines.append('        self.metadata = {"title": "", "authors": [], "year": None}\n')
        new_lines.append('        self._extract_metadata()\n')
        new_lines.append('\n')
    else:
        new_lines.append(line)

with open('pdf_parser.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('fixed')
