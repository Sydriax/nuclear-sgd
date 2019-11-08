import csv
import numpy as np

BARN_TO_M2 = 1e-28

# Expects a CSV file with first row as headers
def load_materials(filename):
	with open(filename, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		mat_properties = {}
		for i, row in enumerate(reader):
			if i == 0:
				headers = row
			else:
				mat_properties[row[0]] = {}
				for j in range(1,len(row)):
					mat_properties[row[0]][headers[j]] = float(row[j])
		return mat_properties

def make_macroscopic_data(filename, newfilename):
	mat_properties = load_materials(filename)
	macro_properties = {}
	for mat, prop in mat_properties.items():
		# Generate macroscopic properties
		macro_mat_prop = {}
		## fast absorption cross-section
		macro_mat_prop['fast-absorb'] = prop['num_density']*prop['sigma-absorb-fast']*BARN_TO_M2
		## fast fission neutron-production cross-section
		macro_mat_prop['fast-nprod-fission'] = prop['num_density']*prop['sigma-fission-fast']*BARN_TO_M2*prop['nubar-fast']
		## fast scatter-up cross-section
		macro_mat_prop['fast-scatterup'] = prop['num_density']*prop['sigma-scatter-fast']*BARN_TO_M2*(1-prop['p-scatterdown'])
		## fast scatter-down cross-section
		macro_mat_prop['fast-scatterdown'] = prop['num_density']*prop['sigma-scatter-fast']*BARN_TO_M2*prop['p-scatterdown']
		## thermal absorption cross-section
		macro_mat_prop['thermal-absorb'] = prop['num_density']*prop['sigma-absorb-thermal']*BARN_TO_M2
		## thermal fission neutron-production cross-section
		macro_mat_prop['thermal-nprod-fission'] = prop['num_density']*prop['sigma-fission-thermal']*BARN_TO_M2*prop['nubar-thermal']
		## thermal scatter cross-section
		macro_mat_prop['thermal-scatter'] = prop['num_density']*prop['sigma-scatter-thermal']*BARN_TO_M2
		## spontaneous fission neutron-production density
		macro_mat_prop['spont-nprod-fission'] = prop['num_density']*prop['n-spont-fission']
		macro_properties[mat] = macro_mat_prop
	# Make table (sorting mats and attributes alphabetically)
	mats = sorted(macro_properties.keys())
	atts = sorted(next(iter(macro_properties.values())).keys())
	table = [['materials']]
	for a in atts:
		table[0].append(a)
	for m in mats:
		table.append([m])
		for a in atts:
			table[-1].append(str(macro_properties[m][a]))
	# Export to new csv
	with open(newfilename, 'w+') as f:
		csvwriter = csv.writer(f, delimiter=',')
		csvwriter.writerows(table)

make_macroscopic_data('simpledata.csv', 'macro-simpledata.csv')