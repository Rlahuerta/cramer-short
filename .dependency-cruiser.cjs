/** @type {import('dependency-cruiser').IConfiguration} */
module.exports = {
  forbidden: [
    {
      name: 'no-circular',
      severity: 'warn',
      comment: 'Circular dependencies can lead to hard-to-maintain code',
      from: {},
      to: {
        circular: true,
      },
    },
    {
      name: 'no-orphans',
      severity: 'info',
      comment: 'Orphaned files (not imported anywhere)',
      from: {
        orphan: true,
        pathNot: [
          '\\.test\\.',
          '\\.spec\\.',
          'src/index\\.tsx$',
          'src/cli\\.ts$',
          'scripts/',
        ],
      },
      to: {},
    },
    {
      name: 'tools-no-import-cli',
      severity: 'error',
      comment: 'Tools layer must not import from CLI layer',
      from: {
        path: '^src/tools/',
      },
      to: {
        path: '^src/(cli|components|controllers)/',
      },
    },
    {
      name: 'agent-no-import-cli',
      severity: 'error',
      comment: 'Agent layer must not import from CLI layer',
      from: {
        path: '^src/agent/',
      },
      to: {
        path: '^src/(cli|components|controllers)/',
      },
    },
    {
      name: 'utils-no-import-agent',
      severity: 'warn',
      comment: 'Prefer utils to stay framework-neutral; existing session/export helpers still depend on app types',
      from: {
        path: '^src/utils/',
      },
      to: {
        path: '^src/(agent|tools|cli|components|controllers)/',
      },
    },
  ],
  options: {
    doNotFollow: {
      path: 'node_modules',
    },
    tsPreCompilationDeps: true,
    tsConfig: {
      fileName: './tsconfig.json',
    },
    enhancedResolveOptions: {
      exportsFields: ['exports'],
      conditionNames: ['import', 'require', 'node', 'default'],
    },
    reporterOptions: {
      dot: {
        collapsePattern: 'node_modules/[^/]+',
      },
      archi: {
        collapsePattern: '^(src/[^/]+)',
      },
      text: {
        highlightFocused: true,
      },
    },
  },
};
