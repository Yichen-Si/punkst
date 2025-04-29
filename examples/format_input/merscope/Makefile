# --------- Prepare Vizgen MERSCOPE data

# ─── Configurable parameters ────
RAWDIR	     := {RAWDIR}
COMPRESSED   := {COMPRESSED}
DATADIR      := {DATADIR}

FEATURES     := $(DATADIR)/features.txt
TRANSCRIPTS  := $(DATADIR)/transcripts.tsv
CELLCOORD    := $(DATADIR)/cell_coordinates.tsv
COORDRANGE   := $(DATADIR)/coord_range.txt

TAB := $(shell printf '\t')

ifeq ($(COMPRESSED), 1)
	RAW_METADATA = $(RAWDIR)/cell_metadata.csv.gz
	CAT_CMD = zcat
else
	RAW_METADATA = $(RAWDIR)/cell_metadata.csv
	CAT_CMD = cat
endif

# ─── Step 0. Prepare‐data ─────────────────────────────────────────────────
.PHONY: all
all: $(COORDRANGE) $(FEATURES) $(TRANSCRIPTS)

$(COORDRANGE):
	$(CAT_CMD) $(RAW_METADATA) | cut -d',' -f 4-9 | tail -n +2 | awk -F',' -v OFS="\t" -v out="$(CELLCOORD)" ' \
	    NR==1 {{ \
	        xmin = $$3; ymin = $$4; \
	        xmax = $$5; ymax = $$6; \
	    }} \
	    {{ \
	        if (xmin>$$3) xmin=$$3; \
			if (ymin>$$4) ymin=$$4; \
			if (xmax<$$5) xmax=$$5; \
			if (ymax<$$6) ymax=$$6; \
	        print $$1, $$2 > out; \
	    }} \
	    END {{ \
	        print "XMIN:=" xmin > "$(COORDRANGE)"; \
	        print "XMAX:=" xmax >> "$(COORDRANGE)"; \
	        print "YMIN:=" ymin >> "$(COORDRANGE)"; \
	        print "YMAX:=" ymax >> "$(COORDRANGE)"; \
	    }}'

$(FEATURES):
ifeq ($(COMPRESSED),1)
	@echo "Extracting gene names from compressed data..."
	zcat $(RAWDIR)/cell_by_gene.csv.gz | head -n 1 | sed 's/,/\n/g' | tail -n +2 | grep -v Blank > $@
else
	@echo "Extracting gene names from uncompressed data..."
	head -n 1 $(RAWDIR)/cell_by_gene.csv | sed 's/,/\n/g' | tail -n +2 | grep -v Blank > $@
endif

$(TRANSCRIPTS):
ifeq ($(COMPRESSED),1)
	@echo "Extracting transcripts from compressed data..."
	zcat $(RAWDIR)/detected_transcripts.csv.gz \
	  | cut -d',' -f2-5,9 \
	  | sed \
	      -e '0,/barcode/{{s/barcode/#barcode/}}' \
	      -e 's/,/$(TAB)/g' \
	      -e 's/$$/$(TAB)1/' \
	      -e '0,/barcode/{{s/$(TAB)1$$/$(TAB)count/}}' \
	    > $@
else
	@echo "Extracting transcripts from uncompressed data..."
	cut -d',' -f2-5,9 $(RAWDIR)/detected_transcripts.csv \
	  | sed \
	      -e '0,/barcode/{{s/barcode/#barcode/}}' \
	      -e 's/,/$(TAB)/g' \
	      -e 's/$$/$(TAB)1/' \
	      -e '0,/barcode/{{s/$(TAB)1$$/$(TAB)count/}}' \
	    > $@
endif
