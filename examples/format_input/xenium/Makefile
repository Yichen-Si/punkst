# --------- Prepare 10X Xenium data

# ─── Configurable parameters ────
RAWDIR	     := {RAWDIR}
DATADIR      := {DATADIR}
RAW_TRANSCRIPTS := {RAW_TRANSCRIPTS}

FEATURES     := $(DATADIR)/features.txt
TRANSCRIPTS  := $(DATADIR)/transcripts.tsv
CELLCOORD    := $(DATADIR)/cell_coordinates.tsv
COORDRANGE   := $(DATADIR)/coord_range.txt

TAB := $(shell printf '\t')

# ─── Step 0. Prepare‐data ─────────────────────────────────────────────────
.PHONY: all
all: $(CELLCOORD) $(FEATURES) $(TRANSCRIPTS) $(COORDRANGE)

$(TRANSCRIPTS):
	@echo "Extracting Xenium transcripts → $@"
	zcat $(RAW_TRANSCRIPTS) \
	  | cut -d',' -f4-6 | sed 's/"//g' \
	  | awk -F',' -v OFS="$(TAB)" '{{ print $$2, $$3, $$1, "1" }}' \
	  > $@

# ─── features.txt ← gene names (second column) for Gene Expression assays
$(FEATURES):
	@echo "Extracting Xenium feature list → $@"
	zcat $(RAWDIR)/cell_feature_matrix/features.tsv.gz \
	  | grep "Gene Expression" \
	  | cut -f2 \
	  > $@

# ─── cell_coordinates.tsv + coord_range.txt ← sample all cells
$(CELLCOORD) $(COORDRANGE):
	zcat $(RAWDIR)/cells.csv.gz \
	  | cut -d',' -f2-3 \
	  | tail -n +2 \
	  | awk -F',' \
	    -v OFS="$(TAB)" \
	    -v out="$(CELLCOORD)" \
	    -v range="$(COORDRANGE)" ' \
	  NR==1 {{ \
	    xmin=$$1; xmax=$$1; ymin=$$2; ymax=$$2; \
	    printf "%.4f\t%.4f\n", $$1,$$2 > out; \
	  }} \
	  {{ \
	    if($$1<xmin) xmin=$$1; if($$1>xmax) xmax=$$1; \
	    if($$2<ymin) ymin=$$2; if($$2>ymax) ymax=$$2; \
	    printf "%.4f\t%.4f\n", $$1,$$2 > out; \
	  }} \
	  END {{ \
	    printf "XMIN:=%.4f\n", xmin   > range; \
	    printf "XMAX:=%.4f\n", xmax  >> range; \
	    printf "YMIN:=%.4f\n", ymin  >> range; \
	    printf "YMAX:=%.4f\n", ymax  >> range; \
	  }}'

-include $(COORDRANGE)
