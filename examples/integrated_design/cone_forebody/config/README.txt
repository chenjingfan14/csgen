

The script 'driver.sh' executes the following routines:

clear.sh
	- removes all forebody, diffuser and inlet directories, and their contents

design_vals.py
	- user defines free-stream conditions and design values
		-> these values are exported as JSON for later routines

forebody.py
	- generates conical forebody surface
		-> conical flow field is generated based on free-stream conditions and
		   forebody design values
		-> the conical forebody and shock surfaces are exported as VTK XML
	- generates provisional capture shape
		-> capture angle of singular engine module is calculated from desired
		   smile angle and number of modules
		-> provisional capture shape is designed to capture all air in shock 
		   layer for the calculated capture angle
	- calculates inlet inflow conditions
		-> Coons patch is constructed across provisional capture shape
		-> flow properties are calculated at each point using conical field
		   solution and free-stream conditions
		-> average flow properties and forebody-inlet attachment info is 
		   exported as JSON file

diffuser.py
	- generates truncated Busemann contour
		-> inlet inflow conditions and desired exit pressure are used to 
		   generate Busemann flow field
		-> the field is then truncated at desired truncation angle
		-> the truncated contour is then scaled to accomodate capture shape

run_diffuser.sh
	- runs a puffin simulation for truncated Busemann diffuser

inlet.py
	- corrects provisional capture shape for forebody-inlet integration
		-> inlet shock surface is constructed from puffin simulation
		-> the intersection contour between shock surface and forebody is 
		   calculated
		-> top boundary of the provisional capture shape is replaced by x-y
		   projection of intersection contour
		-> forebody surface is trimmed along intersection contour
	- generates shape-transition inlets
		-> corrected capture shape is streamline-traced forward through 
		   truncated Busemann diffuser to form inlet A
		-> desired exit shape is streamline-traced backward through truncated
		   Busemann diffuser to form inlet B
		-> inlet A and inlet B are blended using chosen blending parameter
		-> blended inlet is translated and rotated to appropriate position
		-> if multiple modules are desired, blended inlet is copied and rotated 