#import "chart.typ": charts-from-json

#set page(width: 11in, height: 8.5in, margin: 0.65in)
#set text(size: 9pt)

#let source = sys.inputs.at("source")

#charts-from-json(source)
