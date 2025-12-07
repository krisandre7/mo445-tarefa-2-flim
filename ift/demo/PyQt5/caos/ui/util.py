
def read_stylesheet(qss_path):
    try:
        qss = open(qss_path, 'r')
        stylesheet = qss.read()

        return stylesheet
    except FileNotFoundError:
        print("Error when trying to open the qss stylesheet file: %s" % qss_path)