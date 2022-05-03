from vtkmodules.vtkCommonDataModel import vtkStructuredGrid
from vtkmodules.vtkCommonCore import vtkPoints, vtkDoubleArray
from vtkmodules.vtkIOXML import vtkXMLStructuredGridWriter

def ndarray_to_vtkxml(array):
	