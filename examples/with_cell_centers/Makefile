# --------- Experimental: using cell centers to promote nuclei specific factors

# ─── Configurable parameters ────
GITPATH      := {GITPATH}
DATADIR      := {DATADIR}
TMPDIR       := {TMPDIR}
TRANSCRIPTS  := {TRANSCRIPTS}
FEATURES     := {FEATURES}
ICOL_X       := {ICOL_X}
ICOL_Y       := {ICOL_Y}
ICOL_FEATURE := {ICOL_FEATURE}
ICOL_COUNT   := {ICOL_COUNT}
SKIP         := {SKIP}
EXCLUDE_FEATURE_REGEX := {EXCLUDE_FEATURE_REGEX}

THREADS      := {THREADS}
SEED         := {SEED}
TILESIZE     := {TILESIZE}
HEXGRIDS     := {HEXGRIDS}
TOPICS       := {TOPICS}
PIXHEX       := {PIXHEX}
NMOVE        := {NMOVE}
RES          := {RES}
SCALE        := {SCALE}

CELLCOORD    := {CELLCOORD}
NUCLEIRADIUS := {NUCLEIRADIUS}

NUR          := $(NUCLEIRADIUS)
PUNKST       := $(GITPATH)/bin/punkst
TILETSV      := $(DATADIR)/transcripts.tiled.tsv
TILEINDEX    := $(DATADIR)/transcripts.tiled.index
COORD_RANGE  := $(DATADIR)/transcripts.tiled.coord_range.tsv
FEATURES     := $(DATADIR)/transcripts.tiled.features.tsv

TAB := $(shell printf '\t')

# ─── Compute the full list of final report files ──────────────────────────
REPORTS := \
  $(foreach H,$(HEXGRIDS),\
    $(foreach K,$(TOPICS),\
      $(DATADIR)/k$(K)_hex_$(H)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES).factor.info.html\
	  $(DATADIR)/k$(K)_hex_$(H)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES).pixel.png\
    )\
  )

.PHONY: all clean
all: $(REPORTS)

# ─── Step 1: bag pixels → tiles.tsv + tiles.index ─────────────────────
$(DATADIR)/prepare.stamp: $(TRANSCRIPTS)
	@echo "[1/6] pts2tiles → $@"
	 $(PUNKST) pts2tiles \
	  --in-tsv $(TRANSCRIPTS) \
	  --icol-x $(ICOL_X) --icol-y $(ICOL_Y) \
	  --icol-feature $(ICOL_FEATURE) --icol-int $(ICOL_COUNT) --skip $(SKIP) \
	  --tile-size $(TILESIZE) \
	  --tile-buffer 1000 \
	  --out-prefix $(DATADIR)/transcripts.tiled \
	  --temp-dir $(TMPDIR) --threads $(THREADS)
	touch $(DATADIR)/prepare.stamp

# ─── Step 2: for each H in HEXGRIDS → hex_H_randomized.txt ────────────
define HEX_RULE
$(DATADIR)/hex_$(1)_$(NUR)_randomized.txt: \
    $(DATADIR)/prepare.stamp \
	$(CELLCOORD)
	@echo "[2/6] tiles2hex (hex=$(1)) → $$@"
	 $(PUNKST) tiles2hex \
	  --in-tsv $(TILETSV) \
	  --in-index $(TILEINDEX) \
	  --feature-dict $(FEATURES) \
	  --anchor-files $(CELLCOORD) \
	  --radius $(NUCLEIRADIUS) \
	  --icol-x $(ICOL_X) --icol-y $(ICOL_Y) \
	  --icol-feature $(ICOL_FEATURE) \
	  --icol-int $(ICOL_COUNT) \
	  --hex-grid-dist $(1) \
	  --min-count 50 \
	  --out $(DATADIR)/hex_$(1)_$(NUR).txt \
	  --temp-dir $(TMPDIR) --threads $(THREADS)
	sort -k1,1 --parallel $(THREADS) \
	  $(DATADIR)/hex_$(1)_$(NUR).txt > $(DATADIR)/hex_$(1)_$(NUR)_randomized.txt
	rm $(DATADIR)/hex_$(1)_$(NUR).txt
endef
$(foreach H,$(HEXGRIDS),$(eval $(call HEX_RULE,$H)))

# ─── Step 3: for each (K,H) → model.tsv + color.rgb.tsv ──────────────
define LDA_MODEL_RULE
$(DATADIR)/k$(2)_hex_$(1)_$(NUR).model.stamp: \
    $(DATADIR)/hex_$(1)_$(NUR)_randomized.txt
	@echo "[3/6] lda4hex (K=$(2), hex=$(1)) → $$@"
	 $(PUNKST) lda4hex \
	  --in-data $(DATADIR)/hex_$(1)_$(NUR)_randomized.txt \
	  --in-meta $(DATADIR)/hex_$(1)_$(NUR).json \
	  --features $(FEATURES) \
	  --n-topics $(2) \
	  --out-prefix $(DATADIR)/k$(2)_hex_$(1)_$(NUR) --transform \
	  --min-count-per-feature 50 --min-count-train 50 --minibatch-size 512 \
	  --exclude-feature-regex $(EXCLUDE_FEATURE_REGEX) \
	  --threads $(THREADS) --seed $(SEED) --n-epochs 1
	touch $(DATADIR)/k$(2)_hex_$(1)_$(NUR).model.stamp
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call LDA_MODEL_RULE,$H,$K))\
  )\
)

define COLOR_RULE
$(DATADIR)/k$(2)_hex_$(1)_$(NUR).color.rgb.tsv: \
    $(DATADIR)/k$(2)_hex_$(1)_$(NUR).model.stamp
	@echo "[3/6] color_helper (K=$(2), hex=$(1)) → $$@"
	python $(GITPATH)/ext/py/color_helper.py \
	  --input $(DATADIR)/k$(2)_hex_$(1)_$(NUR).results.tsv \
	  --output $(DATADIR)/k$(2)_hex_$(1)_$(NUR).color \
	  --skip-columns layer
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call COLOR_RULE,$H,$K)) \
  )\
)

# ─── Step 4: pixel-decode → .tsv + .pseudobulk.tsv ────────────────────
define PIXEL_RULE
$(DATADIR)/k$(2)_hex_$(1)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES).decode.stamp: \
    $(DATADIR)/k$(2)_hex_$(1)_$(NUR).model.stamp\
    $(DATADIR)/prepare.stamp \
	$(CELLCOORD)
	@echo "[4/6] pixel-decode (K=$(2), hex=$(1)) → $$*.tsv"
	 $(PUNKST) pixel-decode \
	  --model $(DATADIR)/k$(2)_hex_$(1)_$(NUR).model.tsv \
	  --in-tsv $(TILETSV) \
	  --in-index $(TILEINDEX) \
	  --anchor $(CELLCOORD) \
	  --out-pref $(DATADIR)/k$(2)_hex_$(1)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES) \
	  --icol-x $(ICOL_X) \
	  --icol-y $(ICOL_Y) \
	  --icol-feature $(ICOL_FEATURE) \
	  --icol-val $(ICOL_COUNT) \
	  --hex-grid-dist $(PIXHEX) \
	  --n-moves $(NMOVE) \
	  --min-init-count 20 \
	  --pixel-res $(RES) \
	  --temp-dir $(TMPDIR) --threads $(THREADS) --seed $(SEED) \
	  --output-original --output-coord-digits 2 --output-prob-digits 4
	touch $(DATADIR)/k$(2)_hex_$(1)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES).decode.stamp
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call PIXEL_RULE,$H,$K))\
  )\
)

# ─── Step 5: draw-pixel-factors → .png ───────────────────────────
define DRAW_RULE
$(DATADIR)/k$(2)_hex_$(1)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES).pixel.png: \
    $(DATADIR)/k$(2)_hex_$(1)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES).decode.stamp \
    $(DATADIR)/k$(2)_hex_$(1)_$(NUR).color.rgb.tsv \
	$(COORDRANGE)
	@echo "[5/6] draw-pixel-factors (K=$(2), hex=$(1)) → $$@"
	 $(PUNKST) draw-pixel-factors \
	  --in-tsv $(DATADIR)/k$(2)_hex_$(1)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES).tsv \
	  --in-color $(DATADIR)/k$(2)_hex_$(1)_$(NUR).color.rgb.tsv \
	  --out $$@ \
	  --range $(COORD_RANGE) --scale $(SCALE)
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call DRAW_RULE,$H,$K))\
  )\
)

# ─── Step 6: de_bulk + factor_report → final .report ───────────────
define DE_RULE
$(DATADIR)/k$(2)_hex_$(1)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES).de_bulk.tsv: \
    $(DATADIR)/k$(2)_hex_$(1)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES).decode.stamp
	@echo "[6/6] de_bulk (K=$(2), hex=$(1)) → $$@"
	python $(GITPATH)/ext/py/de_bulk.py \
	  --input $(DATADIR)/k$(2)_hex_$(1)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES).pseudobulk.tsv \
	  --output $(DATADIR)/k$(2)_hex_$(1)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES).de_bulk.tsv \
	  --feature_label Feature --thread $(THREADS)
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call DE_RULE,$H,$K))\
  )\
)

define REPORT_RULE
$(DATADIR)/k$(2)_hex_$(1)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES).factor.info.html: \
    $(DATADIR)/k$(2)_hex_$(1)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES).decode.stamp \
	$(DATADIR)/k$(2)_hex_$(1)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES).de_bulk.tsv \
    $(DATADIR)/k$(2)_hex_$(1)_$(NUR).color.rgb.tsv
	@echo "[6/6] factor_report (K=$(2), hex=$(1)) → $$@"
	python $(GITPATH)/ext/py/factor_report.py \
	  --de $(DATADIR)/k$(2)_hex_$(1)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES).de_bulk.tsv \
	  --pseudobulk $(DATADIR)/k$(2)_hex_$(1)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES).pseudobulk.tsv \
	  --feature_label Feature \
	  --color_table $(DATADIR)/k$(2)_hex_$(1)_$(NUR).color.rgb.tsv \
	  --output_pref $(DATADIR)/k$(2)_hex_$(1)_$(NUR).$(PIXHEX)_$(NMOVE)_$(RES)
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call REPORT_RULE,$H,$K))\
  )\
)

# ─── Clean up all generated files ──────────────────────────────────
clean:
	rm -rf $(TMPDIR) \
	       $(DATADIR)/transcripts.tiled.* \
	       $(DATADIR)/hex_*_randomized.txt \
	       $(DATADIR)/k*_hex_*
