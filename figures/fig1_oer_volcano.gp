# Gnuplot script for OER Volcano & Scaling Relation
set terminal pdfcairo font "Times,10" size 10in,4in
set output "fig1_oer_gnuplot.pdf"
set multiplot layout 1,2

set title "(a) OER Intermediate Scaling Relation"
set xlabel "Delta G_*OH (eV)"
set ylabel "Delta G_*OOH (eV)"
set grid
plot "oer_data.dat" using 1:2 with points pt 7 ps 1.5 title "MOF Candidates", \
     x + 3.20 with lines dt 2 title "Universal Line (x + 3.20)"

set title "(b) OER Volcano Curve"
set xlabel "Delta G_*O - Delta G_*OH (eV)"
set ylabel "Overpotential eta_OER (V)"
plot "oer_volcano.dat" using 1:2 with points pt 7 ps 1.5 title "MOFs"

unset multiplot
