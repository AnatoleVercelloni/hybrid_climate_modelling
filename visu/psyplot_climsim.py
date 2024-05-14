from tonetcdf import plot_ncdf
import psyplot.project as psy
import psyplot
from psy_maps.plotters import FieldPlotter



print("import ok")

ps_data = psyplot.open_dataset("daily_output_native.nc", decode_times=False)
T_map = ps_data.isel(time_counter = 100).psy.plot.mapplot(name = 'T500', cmap = 'rainbow')
T_map.docs('datagrid')
T_map.update(datagrid={'c': 'k', 'lw': 0.1})
T_map.show()
T_map.save_project("psyplot_out.png")
