import svelte from 'rollup-plugin-svelte'
import resolve from '@rollup/plugin-node-resolve'
import commonjs from '@rollup/plugin-commonjs'
import { terser } from 'rollup-plugin-terser'
import omt from "@surma/rollup-plugin-off-main-thread"
import ignore from 'rollup-plugin-ignore'
import visualizer from 'rollup-plugin-visualizer'
import postcss from 'rollup-plugin-postcss'
import replace from '@rollup/plugin-replace'
import html from '@rollup/plugin-html'
import { readFileSync } from 'fs'
import MagicString from 'magic-string'

const production = !process.env.ROLLUP_WATCH

// Remove a file from the bundle, remembering its contents
function steal({name, assert}) {
  let set_code
  const code = new Promise((resolve, reject) => set_code = resolve)

  const plugin = {
    name: 'steal',
    renderChunk(code, chunk) {
      if (assert) return
      if (chunk.fileName.endsWith(name))
        set_code(code)
    },
    generateBundle(options, bundle) {
      for (const path of Object.keys(bundle))
        if (path.endsWith(name)) {
          if (assert) set_code(bundle[path].source)
          delete bundle[path]
        }
    },
  }

  return [plugin, code]
}

// @rollup/plugin-replace, but allow asynchronous replacements
function async_replace(pattern, sub) {
  async function replace(code) {
    const matches = Array.from(code.matchAll(pattern))
    if (matches.length) {
      const magic = new MagicString(code)
      for (const m of matches)
        magic.overwrite(m.index, m.index + m[0].length, await sub);
      return {
        code: magic.toString(),
        map: magic.generateMap({hires: true})
      }
    }
  }

  return {
    name: 'replace',
    renderChunk: (code, chunk) => replace(code),
    transform: (code, id) => replace(code),
  }
}

// Inline mid_worker.js into main.js as a data url
const [steal_worker, worker] = steal({name: 'mid_worker.js'})
const replace_worker = async_replace(/\"\.\/mid_worker\.js\"/, worker.then(c =>
  "'data:application/javascript," + encodeURIComponent(c).replace(/\'/g, "\\'") + "'"))

// Bundle css and js into index.html
const [steal_js, js] = steal({name: 'main.js'})
const [steal_css, css] = steal({name: 'main.css', assert: true})
const write_html = html({template: async ({ attributes, bundle, files, publicPath, title }) => {
  const template = readFileSync('src/index.html', 'utf8')
  return template.replace(/\$style/, await css)
                 .replace(/\$script/, await js)
}})

// Rollup configuration
export default {
  input: 'src/main.js',
  output: {
    sourcemap: true,
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

    // Inline mid_worker.js into main.js
    steal_worker,
    replace_worker,

    // If we're building for production (npm run build
    // instead of npm run dev), minify
    production && terser(),

    visualizer({
      filename: 'build/stats.html',
      sourcemap: true,
      template: 'treemap',
    }),

    // Inline everything into index.html
    steal_js,
    steal_css,
    write_html,

    // In dev mode, call `npm run start` once
    // the bundle has been generated
    !production && serve(),
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
