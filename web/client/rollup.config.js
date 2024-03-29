import svelte from 'rollup-plugin-svelte'
import resolve from '@rollup/plugin-node-resolve'
import commonjs from '@rollup/plugin-commonjs'
import terser from '@rollup/plugin-terser'
import omt from "@surma/rollup-plugin-off-main-thread"
import ignore from 'rollup-plugin-ignore'
import { visualizer } from 'rollup-plugin-visualizer'
import postcss from 'rollup-plugin-postcss'

const production = !process.env.ROLLUP_WATCH

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

    // In dev mode, call `npm run start` once
    // the bundle has been generated
    !production && serve(),

    // If we're building for production (npm run build
    // instead of npm run dev), minify
    production && terser(),

    visualizer({
      filename: 'build/stats.html',
      sourcemap: true,
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
