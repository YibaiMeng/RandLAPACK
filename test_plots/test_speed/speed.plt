set terminal png size 1000,1000
set logscale y

#Set line styles
set style line 1 \
linecolor rgb '#000000' \
linetype 1 linewidth 2.0\
pointtype 7 pointsize 1.0

#Set line styles
set style line 2 \
linecolor rgb '#0080ff' \
linetype 1 linewidth 2.0 \
pointtype 7 pointsize 1.0

#Set line styles
set style line 3 \
linecolor rgb '#ff0000' \
linetype 1 linewidth 2.0 \
pointtype 7 pointsize 1.0

#Set line styles
set style line 4 \
linecolor rgb '#8A2BE2' \
linetype 1 linewidth 2.0 \
pointtype 7 pointsize 1.0

#looping over m
do for [i in "4096 8192 16384"] {
    #looping n
    do for [j in "128 256 512"] {
        #Set in/out files
        fname_in = sprintf("raw_data/test_%s_%s.dat", i, j)
        #fname_in = sprintf("raw_data/test_QR_%s_%s.dat", i, j)
        fname_out = sprintf("plots/test_%s_%s.png", i, j)
        #fname_out = sprintf("plots/test_QR_%s_%s.png", i, j)
        set output fname_out
        set title "{/*1.8 Stabilization Speed Tests}"
        #set xlabel "x"
        unset xtics
        set ylabel "{/*1.8 Runtime (μs)}"

        # GEQR Commented out
        plot fname_in u 0:1:xtic(8) with linespoints linestyle 1 title "CholQRQ", '' u 0:2:xtic(8) with linespoints linestyle 2 title "PLUL", '' u 0:3:xtic(8) with linespoints linestyle 3 title "HQRQ"#,  '' u 0:4:xtic(8) with linespoints linestyle 4 title "GEQR"
        
        # Check only HQR vs GEQR
        #plot fname_in u 0:3:xtic(8) with linespoints linestyle 3 title "HQRQ",  '' u 0:4:xtic(8) with linespoints linestyle 4 title "GEQR"
    }
}