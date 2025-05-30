scratchpad.pdf: scratchpad.dvi
	dvipdf $< $@

scratchpad.dvi: scratchpad.tex scratchpad.bib
	latex scratchpad
	bibtex scratchpad
	latex scratchpad
	latex scratchpad

clean:
	/bin/rm -f \
		comment.cut \
		scratchpad.aux \
		scratchpad.bbl \
		scratchpad.blg \
		scratchpad.dvi \
		scratchpad.log \
		scratchpad.out \
		scratchpad.pdf \
		scratchpad.xcp
