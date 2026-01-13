'use client';

import { useState } from 'react';
import { signIn } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Eye, Loader2 } from 'lucide-react';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const res = await signIn('credentials', {
        redirect: false,
        email,
        password,
      });

      if (res?.error) {
        setError('Invalid credentials. Please try again.');
        setIsLoading(false);
      } else {
        // Force hard redirect to ensure middleware picks up the session
        window.location.href = '/dashboard';
      }
    } catch (err) {
      setError('An unexpected error occurred.');
      setIsLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen bg-slate-900">
      {/* Left Side - Hero / Branding */}
      <div className="hidden lg:flex flex-col justify-center items-center w-1/2 p-12 bg-gradient-to-br from-blue-900 to-slate-900 text-white relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-full opacity-10 pointer-events-none">
             <div className="absolute top-[-20%] left-[-20%] w-[80%] h-[80%] bg-blue-500 rounded-full blur-[150px]"></div>
             <div className="absolute bottom-[-20%] right-[-20%] w-[60%] h-[60%] bg-teal-500 rounded-full blur-[150px]"></div>
        </div>
        
        <div className="z-10 text-center">
            <div className="mb-6 flex justify-center">
                <div className="h-20 w-20 bg-blue-600 rounded-2xl flex items-center justify-center shadow-lg shadow-blue-500/20">
                    <Eye className="h-10 w-10 text-white" />
                </div>
            </div>
            <h1 className="text-5xl font-bold mb-4 tracking-tight">OptiRetina</h1>
            <p className="text-lg text-blue-100 max-w-md mx-auto leading-relaxed">
                Empowering early diagnosis of Diabetic Retinopathy with state-of-the-art AI.
            </p>
        </div>
      </div>

      {/* Right Side - Login Form */}
      <div className="flex-1 flex items-center justify-center p-8 bg-slate-50 dark:bg-slate-900">
        <Card className="w-full max-w-md shadow-2xl border-0 bg-white/80 backdrop-blur-sm">
          <CardHeader className="space-y-1">
            <CardTitle className="text-2xl font-bold text-center text-slate-900">Sign In</CardTitle>
            <CardDescription className="text-center text-slate-500">
              Enter your credentials to access the portal
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleLogin} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="email">Email Address</Label>
                <Input 
                  id="email" 
                  placeholder="name@hospital.com" 
                  value={email} 
                  onChange={(e) => setEmail(e.target.value)}
                  type="email"
                  required
                  className="bg-white"
                />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                    <Label htmlFor="password">Password</Label>
                    <a href="#" className="text-xs text-blue-600 hover:text-blue-500">Forgot password?</a>
                </div>
                <Input 
                  id="password" 
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="bg-white"
                />
              </div>
              {error && (
                <div className="p-3 bg-red-50 text-red-600 text-sm rounded-md border border-red-100 flex items-center gap-2">
                    <span className="h-1.5 w-1.5 rounded-full bg-red-500"></span>
                    {error}
                </div>
              )}
              
              <Button className="w-full bg-blue-700 hover:bg-blue-800 transition-all duration-200 py-6" type="submit" disabled={isLoading}>
                {isLoading ? (
                    <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Authenticating...
                    </>
                ) : "Access Dashboard"}
              </Button>
            </form>
          </CardContent>
          <CardFooter className="flex justify-center border-t p-6 bg-slate-50/50">
            <p className="text-xs text-slate-500">
              Test Account: <span className="font-mono text-slate-700 bg-slate-200 px-1 rounded">admin@optiretina.com</span> / <span className="font-mono text-slate-700 bg-slate-200 px-1 rounded">password</span>
            </p>
          </CardFooter>
        </Card>
      </div>
    </div>
  );
}
