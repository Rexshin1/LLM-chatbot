import React, { useState } from 'react';

export default function LoginOnboarding({ onLogin }) {
  const [view, setView] = useState('welcome'); // welcome, login, signup
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingGoogle, setLoadingGoogle] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    setError('');

    if (!email || !password) {
      setError('Please fill in all fields.');
      return;
    }

    if (view === 'signup' && !name) {
      setError('Please fill in your name.');
      return;
    }

    setLoading(true);

    // Mock authentication API delay
    setTimeout(() => {
      setLoading(false);
      onLogin();
    }, 1200);
  };

  const handleGoogleLogin = () => {
    setError('');
    setLoadingGoogle(true);

    // Mock Google OAuth sign-in pop-up experience
    setTimeout(() => {
      setLoadingGoogle(false);
      onLogin();
    }, 1500);
  };

  return (
    <div className="login-page-container">
      {loadingGoogle && (
        <div className="google-modal-overlay">
          <div className="google-loading-card">
            <svg className="google-spinner" width="40" height="40" viewBox="0 0 50 50">
              <circle cx="25" cy="25" r="20" fill="none" stroke="var(--primary)" strokeWidth="4" />
            </svg>
            <h3>Signing in with Google...</h3>
            <p>Connecting securely to account.google.com</p>
          </div>
        </div>
      )}

      <div className="login-card-wrapper">
        <div className="login-logo-header">
          <svg width="48" height="48" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M25 48V75L41 59L25 48Z" fill="#7698F9" />
            <path d="M52 60L77 77H52V60Z" fill="#3F6CD8" />
            <path d="M25 35H55C64.665 35 72.5 42.835 72.5 52.5C72.5 62.165 64.665 70 55 70H50" stroke="#5B86F7" strokeWidth="12" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="50" cy="70" r="8" fill="#5B86F7" />
          </svg>
          <h1>REXA</h1>
        </div>

        {view === 'welcome' && (
          <div className="welcome-step fade-in">
            <h2>Welcome to REXA</h2>
            <p>Log in with your account to continue</p>
            <div className="welcome-btn-group">
              <button 
                type="button" 
                onClick={() => setView('login')} 
                className="btn-auth-primary"
              >
                Log in
              </button>
              <button 
                type="button" 
                onClick={() => setView('signup')} 
                className="btn-auth-secondary"
              >
                Sign up
              </button>

              <div className="auth-divider">
                <span>OR</span>
              </div>

              <button 
                type="button" 
                onClick={handleGoogleLogin} 
                className="btn-google-auth"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" style={{marginRight: '8px'}}>
                  <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
                  <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
                  <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" fill="#FBBC05" />
                  <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" fill="#EA4335" />
                </svg>
                Continue with Google
              </button>
            </div>
          </div>
        )}

        {(view === 'login' || view === 'signup') && (
          <form onSubmit={handleSubmit} className="auth-form fade-in">
            <h2>{view === 'login' ? 'Welcome back' : 'Create your account'}</h2>
            
            {error && <div className="auth-error-banner">⚠️ {error}</div>}

            {view === 'signup' && (
              <div className="auth-input-group">
                <label htmlFor="name-input">Full Name</label>
                <input 
                  type="text" 
                  id="name-input"
                  value={name} 
                  onChange={(e) => setName(e.target.value)} 
                  placeholder="John Doe"
                  autoFocus
                  disabled={loading}
                />
              </div>
            )}

            <div className="auth-input-group">
              <label htmlFor="email-input">Email address</label>
              <input 
                type="email" 
                id="email-input"
                value={email} 
                onChange={(e) => setEmail(e.target.value)} 
                placeholder="name@domain.com"
                autoFocus={view === 'login'}
                disabled={loading}
              />
            </div>

            <div className="auth-input-group">
              <label htmlFor="password-input">Password</label>
              <input 
                type="password" 
                id="password-input"
                value={password} 
                onChange={(e) => setPassword(e.target.value)} 
                placeholder="••••••••"
                disabled={loading}
              />
            </div>

            <button 
              type="submit" 
              className="btn-auth-primary submit-btn" 
              disabled={loading}
            >
              {loading ? (
                <div className="auth-spinner" />
              ) : (
                'Continue'
              )}
            </button>

            <div className="auth-divider">
              <span>OR</span>
            </div>

            <button 
              type="button" 
              onClick={handleGoogleLogin} 
              className="btn-google-auth"
              disabled={loading}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" style={{marginRight: '8px'}}>
                <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
                <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
                <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" fill="#FBBC05" />
                <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" fill="#EA4335" />
              </svg>
              Continue with Google
            </button>

            <p className="auth-switch-text">
              {view === 'login' ? (
                <>
                  Don't have an account?{' '}
                  <span onClick={() => { setView('signup'); setError(''); }}>Sign up</span>
                </>
              ) : (
                <>
                  Already have an account?{' '}
                  <span onClick={() => { setView('login'); setError(''); }}>Log in</span>
                </>
              )}
            </p>
            
            <button 
              type="button" 
              onClick={() => { setView('welcome'); setError(''); }} 
              className="btn-auth-back"
              disabled={loading}
            >
              ← Back
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
