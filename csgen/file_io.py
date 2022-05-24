from vtkmodules.vtkCommonDataModel import vtkStructuredGrid
from vtkmodules.vtkCommonCore import vtkPoints, vtkDoubleArray
from vtkmodules.vtkIOXML import vtkXMLStructuredGridWriter
import csv

def ndarray_to_vtkxml(array):
    pass

def coords_to_csv(coords, file_name, **kwargs):
    # check dimensionality of coordinates
    dim = len(coords[0])

    if dim == 2:
        header = kwargs.get('header', ['x', 'y'])
    elif dim == 3:
        header = kwargs.get('header', ['x', 'y', 'z'])
    else:
        raise AssertionError('Coordinates have invalid dimension.')

    with open(file_name + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(header)
        writer.writerows(coords)