#let (chart, chart-colors, chart-from-json, charts-from-json, lq, scatter-chart) = {
  import "@preview/lilaq:0.6.0" as lq

  let golden-ratio = 1.61803398875

  let default-palette = (
    base: white,
    text: black,
    subtext0: luma(35%),
    overlay0: luma(65%),
    surface1: luma(88%),
    surface0: luma(94%),
    peach: rgb("#fe640b"),
    blue: rgb("#1e66f5"),
    green: rgb("#40a02b"),
    mauve: rgb("#8839ef"),
    red: rgb("#d20f39"),
    teal: rgb("#179299"),
    sky: rgb("#04a5e5"),
    lavender: rgb("#7287fd"),
    dark-peach: rgb("#d56a2d"),
    dark-blue: rgb("#2c6db2"),
    dark-green: rgb("#3d7c42"),
    dark-mauve: rgb("#8d4ea8"),
    dark-red: rgb("#b5453f"),
    dark-teal: rgb("#2d8176"),
    dark-sky: rgb("#3c7e9e"),
    dark-lavender: rgb("#6466b4"),
  )

  let chart-color-keys = (
    "peach",
    "blue",
    "green",
    "mauve",
    "red",
    "teal",
    "sky",
    "lavender",
  )

  let dark-chart-color-keys = chart-color-keys.map(key => "dark-" + key)

  let chart-colors(palette, dark: false) = {
    let keys = if dark { dark-chart-color-keys } else { chart-color-keys }
    keys.map(key => palette.at(key))
  }

  let line-chart-plots(plot-data, colors, theme) = {
    let xs = plot-data.at("iteration")
    let line-plots = plot-data.at("line plots")
    let line-styles = plot-data.at("line styles", default: (:))
    let line-colors = plot-data.at("line colors", default: (:))
    let series = line-plots.keys()
    range(series.len()).map(i => {
      let key = series.at(i)
      lq.plot(
        xs,
        plot-data.at(key),
        label: line-plots.at(key),
        color: if key in line-colors {
          theme.at(line-colors.at(key))
        } else {
          colors.at(calc.rem(i, colors.len()))
        },
        mark: none,
        ..if key in line-styles {
          (stroke: (dash: line-styles.at(key)))
        } else {
          ()
        },
      )
    })
  }

  let bar-chart-plots(plot-data, colors, group-width, fills: (:)) = {
    let xs = plot-data.at("iteration")
    let bar-plots = plot-data.at("bar plots")
    let bar-errors = plot-data.at("bar errors", default: (:))
    let series = bar-plots.keys()
    let bar-width = group-width / series.len()
    range(series.len())
      .map(i => {
        let key = series.at(i)
        let offset = (i - (series.len() - 1) / 2) * bar-width
        let color = colors.at(calc.rem(i, colors.len()))
        let values = plot-data.at(key)
        let bar = lq.bar(
          xs,
          values,
          offset: offset,
          width: bar-width,
          fill: fills.at(key, default: color),
          label: bar-plots.at(key),
        )
        if key in bar-errors {
          (
            bar,
            lq.plot(
              xs.map(x => x + offset),
              values,
              yerr: bar-errors.at(key),
              mark: none,
              stroke: none,
              label: none,
            ),
          )
        } else {
          (bar,)
        }
      })
      .flatten()
  }

  let chart(
    plot-title,
    plot-data,
    width: 100%,
    aspect-ratio: golden-ratio,
    legend-position: top + right,
    legend-options: (:),
    show-legend: true,
    palette: none,
    font-size: 1em,
    // Fraction of each category's slot that its bars fill.
    bar-width: 0.8,
    bar-fills: (:),
    annotations: (),
    // Leave half a category slot beside the outer bars by default.
    xlim: auto,
    ylim: auto,
    yaxis: (subticks: none, tick-args: (density: 60%)),
  ) = {
    let theme = if palette == none { default-palette } else { palette }
    let is-bar-chart = "bar plots" in plot-data
    let colors = chart-colors(theme, dark: not is-bar-chart)
    let x-labels = plot-data.at("x labels", default: none)
    let xaxis = if x-labels != none {
      (subticks: none, ticks: plot-data.at("iteration").zip(x-labels))
    } else {
      (subticks: none, tick-args: (density: 60%))
    }

    layout(size => {
      let chart-width = if type(width) == ratio {
        size.width * width
      } else {
        width
      }
      {
        set text(size: font-size, fill: theme.text)
        show lq.selector(lq.tick-label): set text(fill: theme.subtext0)
        show: lq.set-spine(stroke: none)
        show: lq.set-tick(stroke: 0.45pt + theme.overlay0.transparentize(100%))
        show: lq.set-grid(stroke: 0.35pt + theme.surface1)
        show: lq.set-legend(
          fill: theme.base,
          stroke: 0.4pt + theme.surface1,
        )

        lq.diagram(
          width: chart-width,
          height: chart-width / aspect-ratio,
          margin: 0%,
          xlim: if is-bar-chart and xlim == auto {
            (-0.5, plot-data.at("iteration").len() - 0.5)
          } else { xlim },
          ylim: ylim,
          yscale: if not is-bar-chart and plot-title.ends-with("-loss") { "log" } else { "linear" },
          xaxis: xaxis,
          yaxis: yaxis,
          legend: if show-legend { (position: legend-position) + legend-options } else { none },
          grid: (:),
          fill: theme.base,
          ..if is-bar-chart {
            bar-chart-plots(plot-data, colors, bar-width, fills: bar-fills)
          } else {
            line-chart-plots(plot-data, colors, theme)
          },
          ..annotations,
        )
      }
    })
  }

  let scatter-chart(
    plot-data,
    width: 100%,
    aspect-ratio: 1,
    legend-position: top + left,
    show-legend: true,
    palette: none,
    font-size: 1em,
  ) = {
    let theme = if palette == none { default-palette } else { palette }
    let colors = chart-colors(theme)
    let series = plot-data.at("scatter series")
    let names = series.keys()
    let lo = plot-data.at("min")
    let hi = plot-data.at("max")

    layout(size => {
      let chart-width = if type(width) == ratio {
        size.width * width
      } else {
        width
      }
      {
        set text(size: font-size, fill: theme.text)
        show lq.selector(lq.tick-label): set text(fill: theme.subtext0)
        show: lq.set-spine(stroke: none)
        show: lq.set-tick(stroke: 0.45pt + theme.overlay0.transparentize(100%))
        show: lq.set-grid(stroke: 0.35pt + theme.surface1)
        show: lq.set-legend(
          fill: theme.base,
          stroke: 0.4pt + theme.surface1,
        )

        lq.diagram(
          width: chart-width,
          height: chart-width / aspect-ratio,
          margin: 5%,
          xscale: "log",
          yscale: "log",
          xaxis: (subticks: none, tick-args: (density: 60%)),
          yaxis: (subticks: none, tick-args: (density: 60%)),
          xlim: (lo, hi),
          ylim: (lo, hi),
          legend: if show-legend { (position: legend-position) } else { none },
          grid: (:),
          fill: theme.base,
          lq.plot(
            (lo, hi),
            (lo, hi),
            mark: none,
            color: theme.subtext0,
            stroke: (dash: "dashed", thickness: 0.5pt),
            label: $ y = x $,
          ),
          ..range(names.len()).map(i => {
            let name = names.at(i)
            let pts = series.at(name)
            lq.plot(
              pts.x,
              pts.y,
              mark: "o",
              stroke: none,
              color: colors.at(calc.rem(i, colors.len())),
              label: name,
            )
          }),
        )
      }
    })
  }

  let chart-from-json(
    source,
    plot-key,
    width: 100%,
    aspect-ratio: golden-ratio,
    legend-position: top + right,
    palette: none,
  ) = {
    let data = json(source)
    chart(
      plot-key,
      data.at(plot-key),
      width: width,
      aspect-ratio: aspect-ratio,
      legend-position: legend-position,
      palette: palette,
    )
  }

  let charts-from-json(
    source,
    width: 100%,
    aspect-ratio: golden-ratio,
    legend-position: top + right,
    palette: none,
  ) = {
    let data = json(source)
    for plot-key in data.keys() [
      #align(
        center,
        chart(
          plot-key,
          data.at(plot-key),
          width: width,
          aspect-ratio: aspect-ratio,
          legend-position: legend-position,
          palette: palette,
        ),
      )
      #pagebreak(weak: true)
    ]
  }

  // Exports (fold)
  (
    chart,
    chart-colors,
    chart-from-json,
    charts-from-json,
    lq,
    scatter-chart,
  ) // (end)
}
