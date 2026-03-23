import zipfile
import xml.etree.ElementTree as ET
import sys

def extract_text_from_docx(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as docx:
            content = docx.read('word/document.xml')
        tree = ET.fromstring(content)
        namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        texts = []
        for node in tree.iterfind('.//w:t', namespaces):
            if node.text:
                texts.append(node.text)
        return ''.join(texts)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    text = extract_text_from_docx(sys.argv[1])
    with open('d:/ProLog/docs/extracted_docx.txt', 'w', encoding='utf-8') as f:
        f.write(text)
