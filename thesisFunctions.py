# The following are functions specifically for my thesis and data, rather than generalizable functions as in the
# mlutilities library.

def createDescriptionFromFileName(fileName):
    """
    Takes in a file name (without any directory path) and turns it into a pretty string
    :param fileName:
    :return:
    """
    fileNameWithoutExtension = fileName.split('.')[0]
    fileNamePieces = fileNameWithoutExtension.split('_')

    capitalizedFileNamePieces = []
    for fileNamePiece in fileNamePieces:
        firstLetter = fileNamePiece[0]
        theRest = fileNamePiece[1:]
        firstLetter = firstLetter.capitalize()
        capitalizedFileNamePieces.append(firstLetter + theRest)

    prettyDescription = ' '.join(capitalizedFileNamePieces)
    return prettyDescription
