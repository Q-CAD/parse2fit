import sys
import re
import xml.etree.ElementTree as ET
import numpy as np

class XMLParser:
    def __init__(self, xml_path, false_root=True):
        if false_root is False:
            self.root = ET.parse(xml_path).getroot()
        else:
            with open(xml_path) as f:
                xml = f.read()
            self.root = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")

    def evaluate_type(self, val, val_type):
        # Add array-parsing logic here
        if val_type == "string":
            return val
        elif val_type == 'float' or val_type == 'int' or val_type == 'boolean':
            return eval(val)
        elif val_type == 'array':
            return [eval(f) for f in val.split()]
        else:
            return val

    def find_by_elements(self, start, tags: list=[], attribs: list=[]):
        # Method to recursively find elements below list of element tags
        # Can optionally pass attributes (list of dictionary objects) to check for at each level

        if not tags:
            return start # Return the element if no tags remaining

        current_tag = tags[0]
        current_dct = attribs[0]
        elements = []
        check_elements = start.findall(current_tag)
        if check_elements:
            if current_dct is not None and type(current_dct) == dict: # Choose from criteria
                key = list(current_dct.keys())[0]
                value = current_dct[key]
                for el in check_elements:
                    if self.check_element_attrib(el, key, value):
                        elements.append(el)
            else:
                elements = check_elements
        remaining_tags = tags[1:]
        remaining_attribs = attribs[1:]

        if len(remaining_tags) == 0:
            return elements
        else:
            return self.find_by_elements(elements[0], remaining_tags, remaining_attribs)
        return None

    def check_element_attrib(self, element, key, value):
        if key in element.attrib and element.attrib[key] == value:
            return True
        else:
            return False

    def get_element_attrib(self, element, attribute):
        if attribute in element.attrib:
            return element.attrib[attribute]
        else:
            return None

    def get_formatted_element_text(self, element, val_type=None):
        #val_type = self.get_element_attrib(element, 'type')
        if val_type == None: # Non-specific value parsing
            try:
                return eval(element.text)
            except NameError:
                return element.text
        else:
            return self.evaluate_type(element.text, val_type)
