class SingleToolBase:
    def __init__(self):
        self.tool_name = ""
        self.brief_description = ""
        self.detailed_description = ""

    def run(self):
        raise NotImplementedError("Subclasses must implement run() method.")

    def get_doc(self):
        if self.detailed_description != None and self.detailed_description != "":
            return self.detailed_description
        else:
            return self.run.__doc__

class MultiToolBase:
    def __init__(self):
        self.tools_name = ""
        self.tools_list = ""
        self.brief_description = ""
        self.detailed_description = ""

    def get_doc(self):
        if self.detailed_description != None and self.detailed_description != "":
            return self.detailed_description
        else:
            return self.__doc__