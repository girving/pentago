import svelte from 'rollup-plugin-svelte'
import resolve from '@rollup/plugin-node-resolve'
import commonjs from '@rollup/plugin-commonjs'
import { terser } from 'rollup-plugin-terser'
import omt from "@surma/rollup-plugin-off-main-thread"
import ignore from 'rollup-plugin-ignore'
import visualizer from 'rollup-plugin-visualizer'
import postcss from 'rollup-plugin-postcss'
import replace from '@rollup/plugin-replace'
import { readFileSync } from 'fs'

const production = !process.env.ROLLUP_WATCH

// Inline mid_worker.js into main.js as a data url
function inline_worker() {
  let set_code
  const code = new Promise((resolve, reject) => set_code = resolve)
  const url = code.then(c => "'data:application/javascript," + encodeURIComponent(c).replace(/\'/g, "\\'") + "'")

  return {
    name: 'inline-worker',
    renderChunk(code, chunk) {
      if (chunk.fileName.endsWith('mid_worker.js'))
        set_code(code)
      else if (chunk.fileName.endsWith('main.js'))
        return url.then(u => code.replace(/"\.\/mid_worker\.js"/, u))
    },
    generateBundle(options, bundle) {
      // Drop mid_worker.js from output
      for (const path of Object.keys(bundle))
        if (path.endsWith('mid_worker.js'))
          delete bundle[path]
    },
  }
}

export default {
  input: 'src/main.js',
  output: {
    sourcemap: false,
    format: 'esm',
    name: 'app',
    dir: 'public',
    chunkFileNames: '[name].js'
  },
  plugins: [
    svelte({
      dev: !production,
      emitCss: true,
    }),
    postcss({
      extract: true,
      minimize: production,
    }),
    ignore(['fs']),
    resolve({browser: true, dedupe: ['svelte']}),
    commonjs(),
    omt(),

    // Inline mid.wasm into mid_worker.js
    replace({"fs.readFileSync('../build/mid.wasm')": () => {
      const hex = readFileSync('build/mid.wasm').toString('hex')
      return 'new Uint8Array("' + hex + '".match(/../g).map(n => parseInt(n, 16)))'
    }, delimiters: ['', '']}),

    // If we're building for production (npm run build
    // instead of npm run dev), minify
    production && terser(),

    inline_worker(),

    // In dev mode, call `npm run start` once
    // the bundle has been generated
    !production && serve(),

    visualizer({
      filename: 'build/stats.html',
      sourcemap: false,
      template: 'treemap',
    }),
  ],
  watch: {
    clearScreen: false
  }
}

function serve() {
  let started = false

  return {
    writeBundle() {
      if (!started) {
        started = true

        require('child_process').spawn('npm', ['run', 'start', '--', '--dev'], {
          stdio: ['ignore', 'inherit', 'inherit'],
          shell: true
        })
      }
    }
  }
}
