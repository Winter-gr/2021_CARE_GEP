const express = require('express')
const app = express()
const path = require("path")
const port = 3030

app.use(express.static(path.join(__dirname, "/client/public")));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '/client/public/index.html'))
})

app.get('/basic', (req, res) => {
  res.send('Hello World!')
})

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})