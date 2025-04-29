# --------- Prepare Vizgen MERSCOPE data

# ─── Configurable parameters ────
RAW_TX   := {RAW_TX}
RAW_META := {RAW_META}
RAW_MTX  := {RAW_MTX}
MICRONS_PER_PIXEL := {MICRONS_PER_PIXEL}
DATADIR      := {DATADIR}

FEATURES     := $(DATADIR)/features.txt
TRANSCRIPTS  := $(DATADIR)/transcripts.tsv
CELLCOORD    := $(DATADIR)/cell_coordinates.tsv
COORDRANGE   := $(DATADIR)/coord_range.txt

TAB := $(shell printf '\t')

# ─── Step 0. Prepare‐data ─────────────────────────────────────────────────
.PHONY: all
all: $(COORDRANGE) $(FEATURES) $(TRANSCRIPTS)

$(COORDRANGE):
	cut -d',' -f 7-8 $(RAW_META) | tail -n +2 | awk -F',' -v OFS="$(TAB)" \
		-v mu=$(MICRONS_PER_PIXEL) \
	    -v out="$(CELLCOORD)" \
	    -v range="$(COORDRANGE)" ' \
	    NR==1 {{ \
	        xmin = $$1; ymin = $$2; \
	        xmax = $$1; ymax = $$2; \
	    }} \
	    {{ \
	        if (xmin>$$1) xmin=$$1; \
			if (ymin>$$2) ymin=$$2; \
			if (xmax<$$1) xmax=$$1; \
			if (ymax<$$2) ymax=$$2; \
	        printf "%.2f\t%.2f\n", mu * $$1, mu * $$2 > out; \
	    }} \
	    END {{ \
			printf "XMIN:=%.4f\n", mu * xmin   > range; \
			printf "XMAX:=%.4f\n", mu * xmax  >> range; \
			printf "YMIN:=%.4f\n", mu * ymin  >> range; \
			printf "YMAX:=%.4f\n", mu * ymax  >> range; \
	    }}'

$(FEATURES):
	@echo "Extracting gene names..."
	head -n 1 $(RAW_MTX) | sed -e 's/"//g' -e 's/,/\n/g' | tail -n +3 | grep -v NegPrb > $@

$(TRANSCRIPTS):
	@echo "Extracting transcripts..."
	  awk -F',' -v mu=$(MICRONS_PER_PIXEL) '\
	  NR==1{{gsub(/"/, "", $$0); print $$3, $$4, $$8, "count", $$7, $$9 }}\
	  NR>1{{gsub(/"/, "", $$8); gsub(/"/, "", $$9); printf "%.2f\t%.2f\t%s\t%d\t%d\t%s\n", mu*$$3, mu*$$4, $$8, 1, $$7, $$9 }} ' $(RAW_TX) > $@
