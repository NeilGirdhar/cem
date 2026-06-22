#import "@preview/lilaq:0.6.0" as lq

#let golden-ratio = 1.61803398875

#let default-palette = (
  base: white,
  text: black,
  subtext0: luma(35%),
  overlay0: luma(65%),
  surface1: luma(88%),
  surface0: luma(94%),
  dark-peach: rgb("#d56a2d"),
  dark-blue: rgb("#2c6db2"),
  dark-green: rgb("#3d7c42"),
  dark-mauve: rgb("#8d4ea8"),
  dark-red: rgb("#b5453f"),
  dark-teal: rgb("#2d8176"),
  dark-sky: rgb("#3c7e9e"),
  dark-lavender: rgb("#6466b4"),
)

#let chart-color-keys = (
  "dark-peach",
  "dark-blue",
  "dark-green",
  "dark-mauve",
  "dark-red",
  "dark-teal",
  "dark-sky",
  "dark-lavender",
)

#let chart-colors(palette) = chart-color-keys.map(key => palette.at(key))

#let chart(
  plot-title,
  plot-data,
  width: 100%,
  aspect-ratio: golden-ratio,
  palette: none,
) = {
  let theme = if palette == none { default-palette } else { palette }
  let colors = chart-colors(theme)
  let xs = plot-data.at("iteration")
  let series = plot-data
    .keys()
    .filter(key => {
      if key == "iteration" {
        return false
      }
      let values = plot-data.at(key)
      type(values) == array and values.len() == xs.len()
    })

  layout(size => {
    let chart-width = if type(width) == ratio {
      size.width * width
    } else {
      width
    }
    {
      set text(fill: theme.text)
      show lq.selector(lq.tick-label): set text(fill: theme.subtext0)
      show: lq.set-spine(stroke: 0.5pt + theme.overlay0)
      show: lq.set-tick(stroke: 0.45pt + theme.overlay0)
      show: lq.set-grid(stroke: 0.35pt + theme.surface1)
      show: lq.set-legend(
        fill: theme.surface0,
        stroke: 0.4pt + theme.surface1,
      )

      lq.diagram(
        width: chart-width,
        height: chart-width / aspect-ratio,
        title: plot-title,
        xlabel: "Iteration",
        ylabel: "Value",
        legend: (position: top + right),
        grid: (:),
        fill: theme.base,
        ..range(series.len()).map(i => {
          let key = series.at(i)
          lq.plot(
            xs,
            plot-data.at(key),
            label: key,
            color: colors.at(calc.rem(i, colors.len())),
            mark: none,
          )
        }),
      )
    }
  })
}

#let chart-from-json(
  source,
  plot-key,
  width: 100%,
  aspect-ratio: golden-ratio,
  palette: none,
) = {
  let data = json(source)
  chart(
    plot-key,
    data.at(plot-key),
    width: width,
    aspect-ratio: aspect-ratio,
    palette: palette,
  )
}

#let charts-from-json(source, width: 100%, aspect-ratio: golden-ratio, palette: none) = {
  let data = json(source)
  for plot-key in data.keys() [
    #align(
      center,
      chart(
        plot-key,
        data.at(plot-key),
        width: width,
        aspect-ratio: aspect-ratio,
        palette: palette,
      ),
    )
    #pagebreak(weak: true)
  ]
}
