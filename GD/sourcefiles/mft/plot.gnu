reset
cd 'C:\Users\Theo Christiaanse\Desktop\2017 - Python model\Regenerator Model - V5\sourcefiles\mft'


##################### Specific Heat Plot ####################

set terminal postscript eps size 11cm,6.7cm enhanced color font "Arial,20" linewidth 2
set output 'cp_Gd.eps'

set style line 11 lw 2 lt 1 lc rgb '#0072bd' pt 2  # blue
set style line 12 lw 2 lt 1 lc rgb '#d95319' pt 5  # orange
set style line 13 lw 2 lt 1 lc rgb '#edb120' pt 7  # yellow
set style line 14 lw 2 lt 1 lc rgb '#7e2f8e' pt 9  # purple

set style line 21 lw 2 lt 3 lc rgb '#0072bd' pt 2  # blue
set style line 22 lw 2 lt 3 lc rgb '#d95319' pt 5  # orange
set style line 23 lw 2 lt 3 lc rgb '#edb120' pt 7  # yellow
set style line 24 lw 2 lt 3 lc rgb '#7e2f8e' pt 9  # purple


set ylabel "Heat Capacity [J/kgK]" offset +1
set xlabel "Temperature"
#set xtics 0.25
set xrange [250:350]
#set yrange [0.95:1.05]
set key outside right 
set key spacing 1.1
set datafile separator "\t"
fileName = "gdcp_py.txt"
plot 	fileName every ::1 using 1:2  with linespoints linestyle 11 title "B=0T",\
        fileName every ::1 using 1:7  with linespoints linestyle 12 title "B=0.5T",\
        fileName every ::1 using 1:13  with linespoints linestyle 13 title "B=1.1T",\
        fileName every ::1 using 1:17 with linespoints linestyle 14 title "B=1.5T"
        
        

        


##################### Magnetization Plot ####################

set terminal postscript eps size 11cm,6.7cm enhanced color font "Arial,20" linewidth 2
set output 'Mag_Gd.eps'

set style line 10 lw 2 lt 1 lc rgb '#FF0000' pt 1  # red
set style line 11 lw 2 lt 1 lc rgb '#0072bd' pt 2  # blue
set style line 12 lw 2 lt 1 lc rgb '#d95319' pt 5  # orange
set style line 13 lw 2 lt 1 lc rgb '#edb120' pt 7  # yellow
set style line 14 lw 2 lt 1 lc rgb '#7e2f8e' pt 9  # purple

set style line 21 lw 2 lt 3 lc rgb '#0072bd' pt 2  # blue
set style line 22 lw 2 lt 3 lc rgb '#d95319' pt 5  # orange
set style line 23 lw 2 lt 3 lc rgb '#edb120' pt 7  # yellow
set style line 24 lw 2 lt 3 lc rgb '#7e2f8e' pt 9  # purple


set ylabel "Magnetization [Am^2/kg]" offset +1
set xlabel "Field [T]"
#set xtics 0.25
set xrange [0:2]
set yrange [0:170]
set key outside right 
set key spacing 1.1
set datafile separator "\t"
fileName = "magTransPosed.txt"
plot    fileName every ::1 using 1:2 with linespoints linestyle 10 title "T=247.2K",\
        fileName every ::1 using 1:3 with linespoints linestyle 11 title "T=277.8K",\
        fileName every ::1 using 1:4 with linespoints linestyle 12 title "T=298.4K",\
        fileName every ::1 using 1:5 with linespoints linestyle 13 title "T=324K",\
        fileName every ::1 using 1:6 with linespoints linestyle 14 title "T=350K"