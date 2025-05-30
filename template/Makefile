template.pdf: template.dvi
	dvipdf $< $@

template.dvi: template.tex template.bib
	latex template
	bibtex template
	latex template
	latex template

clean:
	/bin/rm -f \
		comment.cut \
		template.aux \
		template.bbl \
		template.blg \
		template.dvi \
		template.log \
		template.out \
		template.pdf \
		template.xcp
